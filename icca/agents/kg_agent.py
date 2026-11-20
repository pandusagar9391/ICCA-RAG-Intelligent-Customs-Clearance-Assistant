"""
Knowledge Graph Agent for ICCA-RAG.

Specialized agent responsible for:
- Querying customs entities and relations.
- Multi-hop compliance path reasoning (Product -> HS Code -> Regulations -> Docs -> Rules -> Tariffs).
- Extracting localized subgraphs and generating formatted knowledge triples for LLM context injection.
- Ingesting and updating graph knowledge.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from icca.agents.base_agent import BaseAgent
from icca.config import default_config, ICCAConfig
from icca.knowledge_graph.entity_extractor import CustomsEntityExtractor
from icca.knowledge_graph.graph_store import CustomsGraphStore
from icca.knowledge_graph.schema import CustomsEntity, CustomsRelation
from icca.knowledge_graph.traverser import GraphTraverser

logger = logging.getLogger("KnowledgeGraphAgent")


class KnowledgeGraphAgent(BaseAgent):
    """
    Knowledge Graph Specialist Agent.
    Executes graph queries, multi-hop traversals, entity extractions, and context packaging.
    """

    def __init__(
        self,
        name: str = "KnowledgeGraphAgent",
        role: str = "Knowledge Graph Specialist",
        graph_store: Optional[CustomsGraphStore] = None,
        config: Optional[ICCAConfig] = None
    ):
        super().__init__(name=name, role=role)
        self.config = config or default_config

        # Initialize Graph Store
        if graph_store is not None:
            self.graph_store = graph_store
        else:
            self.graph_store = CustomsGraphStore(name="ICCA_Customs_KG")
            default_kg_file = self.config.kg_dir / "customs_kg.json"
            if default_kg_file.exists():
                try:
                    self.graph_store.load_json(default_kg_file)
                    self.log(f"Pre-loaded customs knowledge graph from {default_kg_file.name}")
                except Exception as e:
                    self.log(f"Warning: Failed to load seed KG from {default_kg_file}: {e}", level=logging.WARNING)

        # Submodules
        self.extractor = CustomsEntityExtractor(llm_provider=self.config.llM_provider)
        self.traverser = GraphTraverser(self.graph_store)

        stats = self.graph_store.get_stats()
        self.log(
            f"Initialized with {stats['total_entities']} entities and "
            f"{stats['total_relations']} relations using {stats['engine']} engine."
        )

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a delegated knowledge graph task.

        Supported actions:
        - 'query_entities': lookup entities by ID, type, or keyword.
        - 'traverse': multi-hop compliance path reasoning for a commodity or code.
        - 'get_context_triples': extracts entities from query and returns formatted context for prompt injection.
        - 'add_knowledge': adds new entities/relations to graph and optionally persists to disk.
        - 'get_stats': returns node, edge, and schema distribution statistics.
        - 'export_graph': exports graph to GML, GraphML, or JSON format.

        Args:
            task: Task dictionary containing 'action' and parameters (either top-level or in 'payload').

        Returns:
            Dictionary with 'status' ('success' or 'error'), 'data', and 'message'.
        """
        action = task.get("action")
        payload = task.get("payload", {})

        if not action:
            return {
                "status": "error",
                "message": "Missing 'action' key in task dictionary.",
                "data": None
            }

        try:
            if action == "query_entities":
                return self._handle_query_entities(task, payload)
            elif action == "traverse":
                return self._handle_traverse(task, payload)
            elif action == "get_context_triples":
                return self._handle_get_context_triples(task, payload)
            elif action == "add_knowledge":
                return self._handle_add_knowledge(task, payload)
            elif action == "get_stats":
                return self._handle_get_stats()
            elif action == "export_graph":
                return self._handle_export_graph(task, payload)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported action '{action}'. Valid actions: query_entities, traverse, get_context_triples, add_knowledge, get_stats, export_graph.",
                    "data": None
                }
        except Exception as e:
            self.log(f"Error executing action '{action}': {str(e)}", level=logging.ERROR)
            return {
                "status": "error",
                "message": f"Failed to execute '{action}': {str(e)}",
                "data": None
            }

    def _get_field(self, task: Dict[str, Any], payload: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Helper to get a field from either the top-level task dict or the payload dict."""
        if key in task and task[key] is not None:
            return task[key]
        return payload.get(key, default)

    def _handle_query_entities(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup entities by ID, type, or keyword search."""
        entity_id = self._get_field(task, payload, "entity_id")
        entity_type = self._get_field(task, payload, "entity_type")
        keyword = self._get_field(task, payload, "keyword")

        results = []
        if entity_id:
            entity = self.graph_store.get_entity(str(entity_id))
            if entity:
                results.append(entity.to_dict())
        elif entity_type:
            entities = self.graph_store.get_entities_by_type(str(entity_type))
            results = [e.to_dict() for e in entities]
        elif keyword:
            entities = self.graph_store.find_entities_by_keyword(str(keyword))
            results = [e.to_dict() for e in entities]
        else:
            # Return all entities if no filter specified (capped at 50)
            all_entities = list(self.graph_store.entities.values())[:50]
            results = [e.to_dict() for e in all_entities]

        return {
            "status": "success",
            "message": f"Retrieved {len(results)} matching entities.",
            "data": {
                "count": len(results),
                "entities": results
            }
        }

    def _handle_traverse(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-hop compliance path traversal."""
        product_or_hscode = self._get_field(task, payload, "product_or_hscode") or self._get_field(task, payload, "query")
        if not product_or_hscode:
            return {
                "status": "error",
                "message": "Missing 'product_or_hscode' or 'query' parameter.",
                "data": None
            }

        target = self._get_field(task, payload, "target_agency_or_doc")
        self.log(f"Tracing compliance path for '{product_or_hscode}' (target: {target})")

        compliance_path = self.traverser.find_compliance_path(
            product_or_hscode=str(product_or_hscode),
            target_agency_or_doc=str(target) if target else None
        )

        return {
            "status": "success",
            "message": f"Found compliance pathway rooted at '{compliance_path.matched_entity_id}'.",
            "data": {
                "path": compliance_path.to_dict(),
                "formatted_text": compliance_path.format_text()
            }
        }

    def _handle_get_context_triples(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant entities from a user query, fetch local k-hop subgraph,
        and format triples into structured prompt text for LLM context injection.
        """
        query = self._get_field(task, payload, "query", "")
        entity_ids = self._get_field(task, payload, "entity_ids", [])
        max_hops = int(self._get_field(task, payload, "max_hops", self.config.kg_max_hops))

        resolved_ids: List[str] = list(entity_ids)

        # If entity_ids were not explicitly provided, extract from query text or keywords
        keywords = self._get_field(task, payload, "keywords", [])
        if not resolved_ids and query:
            extracted = self.extractor.extract_entities(query)
            for e in extracted:
                if e.id in self.graph_store.entities:
                    resolved_ids.append(e.id)
                else:
                    # Fallback lookup in graph store by entity name or ID
                    for m in (self.graph_store.find_entities_by_keyword(e.name) or self.graph_store.find_entities_by_keyword(e.id)):
                        if m.id not in resolved_ids:
                            resolved_ids.append(m.id)

            # If still empty, fall back to keyword matching across query tokens
            if not resolved_ids:
                query_tokens = [w.strip() for w in re.findall(r"\b[A-Za-z0-9_]{4,}\b", query)]
                for tok in query_tokens:
                    for e in self.graph_store.find_entities_by_keyword(tok):
                        if e.id not in resolved_ids:
                            resolved_ids.append(e.id)
                        if len(resolved_ids) >= 5:
                            break
                    if len(resolved_ids) >= 5:
                        break

        if not resolved_ids and keywords:
            for kw in keywords:
                for e in self.graph_store.find_entities_by_keyword(kw):
                    if e.id not in resolved_ids:
                        resolved_ids.append(e.id)
                if len(resolved_ids) >= 5:
                    break

        # Subgraph extraction
        subgraph = self.traverser.get_subgraph_for_entities(
            entity_ids=resolved_ids,
            max_hops=max_hops
        )

        # Format triples into prompt-ready bullet points
        formatted_context = self.traverser.format_triples_for_context(
            triples=subgraph["triples"],
            group_by_subject=True
        )

        triples_list = []
        paths_list = []
        for t in subgraph["triples"]:
            if isinstance(t, (tuple, list)) and len(t) >= 3:
                s, p, o = t[0], t[1], t[2]
            elif isinstance(t, dict):
                s = t.get("subject", t.get("source", ""))
                p = t.get("predicate", t.get("relation", ""))
                o = t.get("object", t.get("target", ""))
            else:
                s, p, o = str(t), "RELATED_TO", ""
            triples_list.append(f"[{s}] --{p}--> [{o}]")
            paths_list.append(f"{s} -> {p} -> {o}")

        return {
            "status": "success",
            "message": f"Generated KG context with {len(subgraph['triples'])} triples.",
            "data": {
                "query": query,
                "seed_entity_ids": resolved_ids,
                "triples": subgraph["triples"],
                "triples_text": triples_list,
                "paths": paths_list,
                "num_triples": len(subgraph["triples"]),
                "nodes": subgraph["nodes"],
                "edges": subgraph["edges"],
                "formatted_context": formatted_context
            }
        }

    def _handle_add_knowledge(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add new entities and relations into the knowledge graph."""
        entities = self._get_field(task, payload, "entities", [])
        relations = self._get_field(task, payload, "relations", [])
        persist = bool(self._get_field(task, payload, "persist", False))

        added_entities_count = 0
        added_relations_count = 0

        for e in entities:
            try:
                self.graph_store.add_entity(e)
                added_entities_count += 1
            except Exception as e_err:
                self.log(f"Failed to add entity: {e_err}", level=logging.WARNING)

        for r in relations:
            try:
                self.graph_store.add_relation(r)
                added_relations_count += 1
            except Exception as r_err:
                self.log(f"Failed to add relation: {r_err}", level=logging.WARNING)

        if persist:
            out_file = self.config.kg_dir / "customs_kg.json"
            self.graph_store.save_json(out_file)
            self.log(f"Persisted updated KG to {out_file.name}")

        return {
            "status": "success",
            "message": f"Added {added_entities_count} entities and {added_relations_count} relations to KG.",
            "data": {
                "added_entities": added_entities_count,
                "added_relations": added_relations_count,
                "persisted": persist,
                "total_entities": len(self.graph_store.entities),
                "total_relations": len(self.graph_store.relations)
            }
        }

    def _handle_get_stats(self) -> Dict[str, Any]:
        """Return knowledge graph size and distribution metrics."""
        stats = self.graph_store.get_stats()
        return {
            "status": "success",
            "message": "Knowledge graph metrics retrieved successfully.",
            "data": stats
        }

    def _handle_export_graph(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Export graph to GML, GraphML, or JSON format."""
        export_format = str(self._get_field(task, payload, "format", "json")).lower()
        file_path_str = self._get_field(task, payload, "path")

        if not file_path_str:
            file_path = self.config.kg_dir / f"customs_kg_export.{export_format}"
        else:
            file_path = Path(file_path_str).resolve()

        if export_format == "json":
            self.graph_store.save_json(file_path)
        elif export_format == "gml":
            self.graph_store.export_gml(file_path)
        elif export_format in ("graphml", "xml"):
            self.graph_store.export_graphml(file_path)
        else:
            return {
                "status": "error",
                "message": f"Unsupported format '{export_format}'. Supported: json, gml, graphml.",
                "data": None
            }

        return {
            "status": "success",
            "message": f"Graph exported to '{file_path.name}' in {export_format.upper()} format.",
            "data": {
                "format": export_format,
                "file_path": str(file_path)
            }
        }
