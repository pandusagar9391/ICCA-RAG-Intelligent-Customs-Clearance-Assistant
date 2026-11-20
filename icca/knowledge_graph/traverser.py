"""
Multi-Hop Customs Knowledge Graph Traverser for ICCA-RAG.

Implements the graph traversal, compliance pathway discovery, and triple
formatting engine for prompt augmentation (Section IV-C of ICCA-RAG paper).

Features:
- `find_compliance_path`: multi-hop traversal from product/HS-code to regulations,
  statutory documents, inspection mandates, and tariff schedules.
- `get_subgraph_for_entities`: k-hop localized subgraph extraction for retrieval queries.
- `format_triples_for_context`: prompt injection formatter turning graph triples into
  structured, concise bullet points for LLM reasoning.
"""

from collections import deque
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from icca.knowledge_graph.graph_store import CustomsGraphStore
from icca.knowledge_graph.schema import (
    CompliancePath,
    CustomsEntity,
    CustomsRelation,
    EntityType,
    RelationType,
)

logger = logging.getLogger(__name__)


class GraphTraverser:
    """
    Reasoning engine performing multi-hop semantic graph traversal over CustomsGraphStore.
    """

    def __init__(self, graph_store: CustomsGraphStore):
        """
        Initialize traverser with a backing knowledge graph store.
        
        Args:
            graph_store: CustomsGraphStore instance.
        """
        self.store = graph_store

    # -------------------------------------------------------------------------
    # 1. Multi-Hop Compliance Path Traversal
    # -------------------------------------------------------------------------

    def find_compliance_path(
        self,
        product_or_hscode: str,
        target_agency_or_doc: Optional[str] = None
    ) -> CompliancePath:
        """
        Trace the end-to-end statutory compliance pathway for a product or HS code.
        Path: Product Category -> HS Code -> Applicable Regulations -> Required Documents -> Inspection Rules & Tariffs.
        
        Args:
            product_or_hscode: Product name, keyword, or HS code (e.g., '8429', 'laptop', 'hazardous chemical').
            target_agency_or_doc: Optional target agency or document ID to check specific reachability.
            
        Returns:
            CompliancePath: Structured report with all connected compliance requirements.
        """
        cleaned_query = product_or_hscode.strip()

        # Step 1: Resolve query to starting entity node(s)
        start_entities = self._resolve_start_entities(cleaned_query)

        if not start_entities:
            # Fallback path when nothing found in graph
            return CompliancePath(
                product_or_hscode=product_or_hscode,
                matched_entity_id=None,
                summary=f"No matching customs entity or HS heading found in knowledge graph for '{product_or_hscode}'."
            )

        primary_entity = start_entities[0]
        primary_id = primary_entity.id

        # Accumulators
        collected_hs: Dict[str, Dict[str, Any]] = {}
        collected_regs: Dict[str, Dict[str, Any]] = {}
        collected_docs: Dict[str, Dict[str, Any]] = {}
        collected_agencies: Dict[str, Dict[str, Any]] = {}
        collected_rules: Dict[str, Dict[str, Any]] = {}
        collected_tariffs: Dict[str, Dict[str, Any]] = {}
        collected_triples: List[Tuple[str, str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        def record_triple(src: str, rel: str, tgt: str):
            t = (src, rel, tgt)
            if t not in seen_triples:
                seen_triples.add(t)
                collected_triples.append(t)

        # Step 2: Expand from start entities to HS Codes if start is a product category
        hs_nodes_to_traverse: List[CustomsEntity] = []

        for entity in start_entities:
            if entity.type == EntityType.HS_CODE.value:
                collected_hs[entity.id] = entity.to_dict()
                hs_nodes_to_traverse.append(entity)

                # Parent HS heading inheritance (e.g. HS_8471.30 inherits from HS_8471)
                if "." in entity.id or ("_" in entity.id and "." in entity.id.split("_")[-1]):
                    raw_code = entity.id.replace("HS_", "")
                    if "." in raw_code:
                        parent_code = raw_code.split(".")[0]
                        parent_id = f"HS_{parent_code}"
                        parent_entity = self.store.get_entity(parent_id)
                        if parent_entity and parent_entity.id not in collected_hs:
                            collected_hs[parent_entity.id] = parent_entity.to_dict()
                            hs_nodes_to_traverse.append(parent_entity)
                            record_triple(entity.id, RelationType.CLASSIFIED_AS.value, parent_entity.id)
            elif entity.type == EntityType.PRODUCT_CATEGORY.value:
                # Traverse CLASSIFIED_AS outgoing edges to find HS codes
                for nbr_id, rel_type, _ in self.store.get_neighbors(entity.id, direction="out"):
                    if rel_type == RelationType.CLASSIFIED_AS.value:
                        nbr_entity = self.store.get_entity(nbr_id)
                        if nbr_entity and nbr_entity.type == EntityType.HS_CODE.value:
                            collected_hs[nbr_id] = nbr_entity.to_dict()
                            hs_nodes_to_traverse.append(nbr_entity)
                            record_triple(entity.id, rel_type, nbr_id)

                # Also traverse directly connected docs/regulations from category
                self._harvest_entity_neighbors(
                    entity.id,
                    collected_docs,
                    collected_regs,
                    collected_rules,
                    collected_tariffs,
                    record_triple
                )

        # If start entities were HS codes, also check incoming CLASSIFIED_AS from categories
        if not collected_hs and start_entities:
            hs_nodes_to_traverse = start_entities

        # Step 3: Traverse outwards from each HS Code node
        for hs in hs_nodes_to_traverse:
            self._harvest_entity_neighbors(
                hs.id,
                collected_docs,
                collected_regs,
                collected_rules,
                collected_tariffs,
                record_triple
            )

        # Step 4: Traverse from collected documents to identify Issuing Agencies
        for doc_id in list(collected_docs.keys()):
            for nbr_id, rel_type, _ in self.store.get_neighbors(doc_id, direction="out"):
                if rel_type == RelationType.ISSUED_BY.value:
                    agency_entity = self.store.get_entity(nbr_id)
                    if agency_entity:
                        collected_agencies[nbr_id] = agency_entity.to_dict()
                        record_triple(doc_id, rel_type, nbr_id)

        # Step 5: Filter or verify target_agency_or_doc if specified
        target_found = None
        if target_agency_or_doc:
            target_norm = target_agency_or_doc.strip().lower()
            for doc_id, doc_dict in collected_docs.items():
                if target_norm in doc_id.lower() or target_norm in doc_dict["name"].lower():
                    target_found = f"Required Document '{doc_dict['name']}' ({doc_id}) is mandatory in this compliance pathway."
                    break
            if not target_found:
                for agency_id, agency_dict in collected_agencies.items():
                    if target_norm in agency_id.lower() or target_norm in agency_dict["name"].lower():
                        target_found = f"Regulatory Agency '{agency_dict['name']}' ({agency_id}) exercises statutory jurisdiction over this pathway."
                        break

        # Step 6: Construct structured summary
        summary = self._synthesize_path_summary(
            product_or_hscode=product_or_hscode,
            primary_entity=primary_entity,
            hs_dict=collected_hs,
            regs_dict=collected_regs,
            docs_dict=collected_docs,
            rules_dict=collected_rules,
            tariffs_dict=collected_tariffs,
            target_status=target_found
        )

        return CompliancePath(
            product_or_hscode=product_or_hscode,
            matched_entity_id=primary_id,
            hs_codes=list(collected_hs.values()),
            regulations=list(collected_regs.values()),
            required_documents=list(collected_docs.values()),
            agencies=list(collected_agencies.values()),
            inspection_rules=list(collected_rules.values()),
            tariff_rates=list(collected_tariffs.values()),
            triples=collected_triples,
            summary=summary
        )

    def _harvest_entity_neighbors(
        self,
        node_id: str,
        collected_docs: Dict[str, Dict[str, Any]],
        collected_regs: Dict[str, Dict[str, Any]],
        collected_rules: Dict[str, Dict[str, Any]],
        collected_tariffs: Dict[str, Dict[str, Any]],
        record_triple_fn
    ):
        """Helper to collect 1-hop outgoing compliance relations."""
        for nbr_id, rel_type, _ in self.store.get_neighbors(node_id, direction="out"):
            nbr = self.store.get_entity(nbr_id)
            if not nbr:
                continue

            record_triple_fn(node_id, rel_type, nbr_id)

            if rel_type == RelationType.REQUIRES_DOC.value or nbr.type == EntityType.REQUIRED_DOCUMENT.value:
                collected_docs[nbr_id] = nbr.to_dict()
            elif rel_type == RelationType.REGULATED_BY.value or nbr.type == EntityType.REGULATION.value:
                collected_regs[nbr_id] = nbr.to_dict()
            elif rel_type == RelationType.SUBJECT_TO.value or nbr.type == EntityType.INSPECTION_RULE.value:
                collected_rules[nbr_id] = nbr.to_dict()
            elif rel_type == RelationType.HAS_TARIFF.value or nbr.type == EntityType.TARIFF_RATE.value:
                collected_tariffs[nbr_id] = nbr.to_dict()

    def _resolve_start_entities(self, query: str) -> List[CustomsEntity]:
        """Resolve query string to the best matching entities in the knowledge graph."""
        query_norm = query.strip()

        # 1. Direct ID match
        direct = self.store.get_entity(query_norm)
        if direct:
            return [direct]

        # 2. HS Code pattern match (e.g., '8429', '8471.30', 'HS 8429')
        hs_match = re.search(r"\b([0-9]{4}(?:\.[0-9]{2}(?:\.[0-9]{2})?)?)\b", query_norm)
        if hs_match:
            raw_code = hs_match.group(1)
            # Try exact ID variations: 'HS_8429', 'HS_8471.30', '8429'
            for prefix in [f"HS_{raw_code}", raw_code, f"HS_{raw_code.split('.')[0]}"]:
                cand = self.store.get_entity(prefix)
                if cand:
                    return [cand]

        # 3. Keyword / Text search across entities
        matched = self.store.find_entities_by_keyword(query_norm)
        if matched:
            # Prioritize HS_CODE and PRODUCT_CATEGORY entities first
            def entity_priority(e: CustomsEntity):
                if e.type == EntityType.HS_CODE.value:
                    return 0
                if e.type == EntityType.PRODUCT_CATEGORY.value:
                    return 1
                return 2

            matched.sort(key=entity_priority)
            return matched

        return []

    def _synthesize_path_summary(
        self,
        product_or_hscode: str,
        primary_entity: CustomsEntity,
        hs_dict: Dict[str, Any],
        regs_dict: Dict[str, Any],
        docs_dict: Dict[str, Any],
        rules_dict: Dict[str, Any],
        tariffs_dict: Dict[str, Any],
        target_status: Optional[str]
    ) -> str:
        """Construct a clear compliance overview text."""
        parts = []
        parts.append(f"Clearance pathway identified for '{product_or_hscode}' rooted at [{primary_entity.id} - {primary_entity.name}].")

        if hs_dict:
            hs_str = ", ".join(f"{h.get('id')} ({h.get('name')})" for h in hs_dict.values())
            parts.append(f"- Relevant HS Classifications: {hs_str}.")

        if docs_dict:
            doc_str = ", ".join(f"{d.get('name')}" for d in docs_dict.values())
            parts.append(f"- Mandatory Clearance Documents: {doc_str}.")

        if regs_dict:
            reg_str = ", ".join(f"{r.get('name')}" for r in regs_dict.values())
            parts.append(f"- Applicable Statutory Frameworks: {reg_str}.")

        if rules_dict:
            rule_str = ", ".join(f"{ru.get('name')}" for ru in rules_dict.values())
            parts.append(f"- Special Inspection Mandates: {rule_str}.")

        if tariffs_dict:
            for t in tariffs_dict.values():
                props = t.get("properties", {})
                parts.append(f"- Duty Schedule: BCD={props.get('basic_customs_duty', 'N/A')}, IGST={props.get('igst', 'N/A')}, SWS={props.get('social_welfare_surcharge', '10%')}.")

        if target_status:
            parts.append(f"- Target Verification: {target_status}")

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # 2. Localized Subgraph Extraction (k-hop BFS)
    # -------------------------------------------------------------------------

    def get_subgraph_for_entities(
        self,
        entity_ids: List[str],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Extract the k-hop neighborhood subgraph surrounding the specified entity IDs.
        
        Args:
            entity_ids: List of source entity IDs.
            max_hops: Traversal depth limit (default 2).
            
        Returns:
            Dictionary with:
            - 'nodes': List of CustomsEntity dictionaries
            - 'edges': List of CustomsRelation dictionaries
            - 'triples': List of tuples (source_id, relation_type, target_id)
        """
        visited_nodes: Set[str] = set()
        visited_edges: Set[Tuple[str, str, str]] = set()

        nodes_list: List[Dict[str, Any]] = []
        edges_list: List[Dict[str, Any]] = []
        triples_list: List[Tuple[str, str, str]] = []

        queue = deque()
        for eid in entity_ids:
            eid_str = str(eid).strip()
            if self.store.get_entity(eid_str):
                queue.append((eid_str, 0))
                visited_nodes.add(eid_str)

        while queue:
            current_id, current_hop = queue.popleft()
            current_entity = self.store.get_entity(current_id)
            if current_entity:
                nodes_list.append(current_entity.to_dict())

            if current_hop >= max_hops:
                continue

            # Traverse outgoing neighbors: current_id -> nbr_id
            for nbr_id, rel_type, edge_data in self.store.get_neighbors(current_id, direction="out"):
                edge_key = (current_id, rel_type, nbr_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    triples_list.append(edge_key)
                    edges_list.append({
                        "source_id": current_id,
                        "target_id": nbr_id,
                        "relation_type": rel_type,
                        "weight": edge_data.get("weight", 1.0),
                        "properties": {k: v for k, v in edge_data.items() if k not in ("relation_type", "weight")},
                    })

                if nbr_id not in visited_nodes:
                    visited_nodes.add(nbr_id)
                    queue.append((nbr_id, current_hop + 1))

            # Traverse incoming neighbors: nbr_id -> current_id
            for nbr_id, rel_type, edge_data in self.store.get_neighbors(current_id, direction="in"):
                edge_key = (nbr_id, rel_type, current_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    triples_list.append(edge_key)
                    edges_list.append({
                        "source_id": nbr_id,
                        "target_id": current_id,
                        "relation_type": rel_type,
                        "weight": edge_data.get("weight", 1.0),
                        "properties": {k: v for k, v in edge_data.items() if k not in ("relation_type", "weight")},
                    })

                if nbr_id not in visited_nodes:
                    visited_nodes.add(nbr_id)
                    queue.append((nbr_id, current_hop + 1))

        return {
            "num_nodes": len(nodes_list),
            "num_edges": len(edges_list),
            "nodes": nodes_list,
            "edges": edges_list,
            "triples": triples_list,
        }

    # -------------------------------------------------------------------------
    # 3. Triple Context Formatter for Prompt Injection
    # -------------------------------------------------------------------------

    def format_triples_for_context(
        self,
        triples: Union[List[Tuple[str, str, str]], List[Dict[str, Any]]],
        group_by_subject: bool = True
    ) -> str:
        """
        Format graph triples into concise, readable bullet points for LLM prompt augmentation.
        
        Example output:
        [HS 8429 (Earth-moving Machinery)] -> REQUIRES_DOC -> [DOC_BILL_OF_ENTRY (Bill of Entry), DOC_CEC (Chartered Engineer Certificate)]
        [HS 8429] -> REGULATED_BY -> [REG_CUSTOMS_ACT_1962 (Customs Act 1962)]
        [HS 8429] -> HAS_TARIFF -> [TARIFF_MACHINERY_DEFAULT (BCD: 7.5%, IGST: 18%)]
        
        Args:
            triples: List of (source_id, relation_type, target_id) tuples or edge dictionaries.
            group_by_subject: Whether to group multiple targets under the same (source, relation).
            
        Returns:
            Formatted string ready for LLM prompt injection.
        """
        if not triples:
            return "No relevant customs knowledge graph triples available."

        # Normalize input to (src, rel, tgt)
        normalized_triples: List[Tuple[str, str, str]] = []
        for item in triples:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                normalized_triples.append((str(item[0]), str(item[1]), str(item[2])))
            elif isinstance(item, dict):
                src = str(item.get("source_id", ""))
                rel = str(item.get("relation_type", item.get("type", "")))
                tgt = str(item.get("target_id", ""))
                if src and rel and tgt:
                    normalized_triples.append((src, rel, tgt))

        if not normalized_triples:
            return "No valid triples found."

        def get_label(entity_id: str) -> str:
            ent = self.store.get_entity(entity_id)
            if ent:
                clean_name = ent.name
                if ent.type == EntityType.TARIFF_RATE.value:
                    bcd = ent.properties.get("basic_customs_duty", "")
                    igst = ent.properties.get("igst", "")
                    if bcd or igst:
                        return f"{clean_name} [BCD: {bcd}, IGST: {igst}]"
                return clean_name
            return entity_id

        if group_by_subject:
            # Group by (source_id, relation_type)
            grouped: Dict[Tuple[str, str], List[str]] = {}
            for src, rel, tgt in normalized_triples:
                key = (src, rel)
                grouped.setdefault(key, []).append(tgt)

            lines = ["Customs Knowledge Graph Context:"]
            for (src, rel), targets in grouped.items():
                src_label = f"[{src}] {get_label(src)}"
                target_labels = [f"[{t}] {get_label(t)}" for t in targets]
                target_str = ", ".join(target_labels)
                lines.append(f"- {src_label} --{rel}--> {target_str}")

            return "\n".join(lines)
        else:
            lines = ["Customs Knowledge Graph Context:"]
            for src, rel, tgt in normalized_triples:
                src_label = f"[{src}] {get_label(src)}"
                tgt_label = f"[{tgt}] {get_label(tgt)}"
                lines.append(f"- {src_label} --{rel}--> {tgt_label}")
            return "\n".join(lines)
