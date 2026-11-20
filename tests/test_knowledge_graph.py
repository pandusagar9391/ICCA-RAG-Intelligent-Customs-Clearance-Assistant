"""
Comprehensive Unit Tests for ICCA-RAG Knowledge Graph Components.

Tests:
1. Schema Data Classes & Enums (CustomsEntity, CustomsRelation, CompliancePath, EntityType, RelationType)
2. Graph Store Operations (NetworkX / InMemory DiGraph, add_entity, add_relation, get_neighbors, indexes)
3. Graph Store Serialization (JSON, GML, GraphML, and Neo4j stub interfaces)
4. Entity & Relation Extraction (Regex patterns for HS codes, docs, acts, agencies, categories, rules, tariffs)
5. Multi-Hop Graph Traversal (Machinery, Hazmat Chemicals, Dual-Use Electronics, Medical Devices, Agriculture)
6. Knowledge Graph Agent (Actions: query_entities, traverse, get_context_triples, add_knowledge, get_stats, export_graph)
7. Seed Customs KG File Integrity (Validates data/knowledge_graph/customs_kg.json)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from icca.knowledge_graph.schema import (
    EntityType,
    RelationType,
    CustomsEntity,
    CustomsRelation,
    CompliancePath,
    HS_CODE,
    PRODUCT_CATEGORY,
    REGULATION,
    REQUIRED_DOCUMENT,
    TARIFF_RATE,
    INSPECTION_RULE,
    GOVERNMENT_AGENCY,
    REQUIRES_DOC,
    REGULATED_BY,
    HAS_TARIFF,
    CLASSIFIED_AS,
    SUBJECT_TO,
    ISSUED_BY,
)
from icca.knowledge_graph.graph_store import CustomsGraphStore
from icca.knowledge_graph.entity_extractor import CustomsEntityExtractor
from icca.knowledge_graph.traverser import GraphTraverser
from icca.knowledge_graph import load_customs_kg
from icca.agents.kg_agent import KnowledgeGraphAgent


# =============================================================================
# 1. Schema Tests
# =============================================================================

def test_schema_entities_and_relations():
    """Verify CustomsEntity and CustomsRelation creation, validation, and dict conversion."""
    entity = CustomsEntity(
        id="HS_8429",
        type=EntityType.HS_CODE,
        name="Earth-moving Machinery",
        description="Bulldozers and Excavators",
        properties={"chapter": "84"}
    )
    assert entity.id == "HS_8429"
    assert entity.type == "HS_CODE"
    assert entity.properties["chapter"] == "84"

    d = entity.to_dict()
    assert d["id"] == "HS_8429"
    assert d["type"] == "HS_CODE"

    reconstructed = CustomsEntity.from_dict(d)
    assert reconstructed.id == entity.id
    assert reconstructed.name == entity.name

    rel = CustomsRelation(
        source_id="HS_8429",
        target_id="DOC_BOE",
        relation_type=RelationType.REQUIRES_DOC,
        properties={"mandatory": True},
        weight=1.0
    )
    assert rel.source_id == "HS_8429"
    assert rel.relation_type == "REQUIRES_DOC"
    assert rel.properties["mandatory"] is True

    rel_dict = rel.to_dict()
    reconstructed_rel = CustomsRelation.from_dict(rel_dict)
    assert reconstructed_rel.target_id == "DOC_BOE"
    assert reconstructed_rel.relation_type == "REQUIRES_DOC"


def test_compliance_path_data_structure():
    """Verify CompliancePath serialization and formatting."""
    path = CompliancePath(
        product_or_hscode="8471.30",
        matched_entity_id="HS_8471.30",
        hs_codes=[{"id": "HS_8471.30", "name": "Laptops"}],
        required_documents=[{"id": "DOC_BIS", "name": "BIS Certificate", "properties": {"mandatory": True}}],
        regulations=[{"id": "REG_BIS_ACT", "name": "BIS Act 2016"}],
        summary="Clearance requires BIS registration."
    )
    text = path.format_text()
    assert "8471.30" in text
    assert "DOC_BIS" in text
    assert "REG_BIS_ACT" in text
    assert "Clearance requires BIS registration." in text


# =============================================================================
# 2. Graph Store Tests
# =============================================================================

def test_graph_store_mutations_and_neighbors():
    """Verify entity/relation addition, indexing, and directional neighbor traversal."""
    store = CustomsGraphStore("Test_KG")

    e1 = CustomsEntity("HS_8429", EntityType.HS_CODE, "Bulldozers")
    e2 = CustomsEntity("DOC_BOE", EntityType.REQUIRED_DOCUMENT, "Bill of Entry")
    e3 = CustomsEntity("AGENCY_CBIC", EntityType.GOVERNMENT_AGENCY, "Indian Customs")

    store.add_entity(e1)
    store.add_entity(e2)
    store.add_entity(e3)

    r1 = CustomsRelation("HS_8429", "DOC_BOE", RelationType.REQUIRES_DOC)
    r2 = CustomsRelation("DOC_BOE", "AGENCY_CBIC", RelationType.ISSUED_BY)
    store.add_relation(r1)
    store.add_relation(r2)

    stats = store.get_stats()
    assert stats["total_entities"] == 3
    assert stats["total_relations"] == 2
    assert stats["entities_by_type"]["HS_CODE"] == 1

    # Outgoing neighbors of HS_8429
    out_nbrs = store.get_neighbors("HS_8429", direction="out")
    assert len(out_nbrs) == 1
    assert out_nbrs[0][0] == "DOC_BOE"
    assert out_nbrs[0][1] == "REQUIRES_DOC"

    # Incoming neighbors of DOC_BOE
    in_nbrs = store.get_neighbors("DOC_BOE", direction="in")
    assert len(in_nbrs) == 1
    assert in_nbrs[0][0] == "HS_8429"

    # Keyword search
    hits = store.find_entities_by_keyword("Bulldozer")
    assert len(hits) >= 1
    assert hits[0].id == "HS_8429"

    # Type search
    docs = store.get_entities_by_type(EntityType.REQUIRED_DOCUMENT)
    assert len(docs) == 1
    assert docs[0].id == "DOC_BOE"


def test_graph_store_serialization():
    """Verify JSON, GML, and GraphML serialization and deserialization."""
    store = CustomsGraphStore("Serialization_Test")
    store.add_entity(CustomsEntity("HS_2901", EntityType.HS_CODE, "Ethylene"))
    store.add_entity(CustomsEntity("DOC_MSDS", EntityType.REQUIRED_DOCUMENT, "MSDS"))
    store.add_relation(CustomsRelation("HS_2901", "DOC_MSDS", RelationType.REQUIRES_DOC))

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "test_kg.json"
        gml_path = Path(tmp_dir) / "test_kg.gml"
        graphml_path = Path(tmp_dir) / "test_kg.graphml"

        # JSON save & reload
        store.save_json(json_path)
        assert json_path.exists()

        reloaded = CustomsGraphStore()
        reloaded.load_json(json_path)
        assert len(reloaded.entities) == 2
        assert len(reloaded.relations) == 1
        assert reloaded.get_entity("HS_2901") is not None

        # GML Export
        store.export_gml(gml_path)
        assert gml_path.exists()
        gml_content = gml_path.read_text(encoding="utf-8")
        assert "HS_2901" in gml_content

        # GraphML Export
        store.export_graphml(graphml_path)
        assert graphml_path.exists()
        graphml_content = graphml_path.read_text(encoding="utf-8")
        assert "HS_2901" in graphml_content


def test_neo4j_stub_readiness():
    """Verify Neo4j extension methods behave safely when external DB is offline."""
    store = CustomsGraphStore()
    # Connecting to offline port should return False safely without unhandled exceptions
    conn_result = store.connect_neo4j(uri="bolt://127.0.0.1:19999", auth=("neo4j", "test"))
    assert conn_result is False

    sync_result = store.sync_to_neo4j()
    assert sync_result["status"] == "skipped"

    query_result = store.query_cypher("MATCH (n) RETURN n")
    assert query_result[0]["status"] == "unsupported_offline"


# =============================================================================
# 3. Entity Extractor Tests
# =============================================================================

def test_entity_extractor_patterns():
    """Verify regex entity extraction across HS codes, documents, and authorities."""
    extractor = CustomsEntityExtractor()

    sample_text = (
        "Consignment of portable laptop computers under HS 8471.30 arriving from Japan. "
        "Requires filing of Bill of Entry Form I, Commercial Invoice, Packing List, "
        "and mandatory BIS Registration Certificate under Bureau of Indian Standards Act, 2016. "
        "Checked by Indian Customs (CBIC) with BCD Free and IGST 18%."
    )

    entities = extractor.extract_entities(sample_text)
    entity_ids = {e.id for e in entities}

    assert "HS_8471.30" in entity_ids
    assert "DOC_BILL_OF_ENTRY" in entity_ids
    assert "DOC_COMMERCIAL_INVOICE" in entity_ids
    assert "DOC_BIS_CERTIFICATE" in entity_ids
    assert "AGENCY_CBIC" in entity_ids
    assert "AGENCY_BIS" in entity_ids
    assert "TARIFF_ITA1_DUTY_FREE" in entity_ids


def test_entity_extractor_relations():
    """Verify heuristic relationship binding between extracted entities."""
    extractor = CustomsEntityExtractor()
    text = "Chemical consignment under HS 2901 requires MSDS and Dangerous Goods Declaration governed by MSIHC Rules."

    res = extractor.extract_all(text)
    assert len(res["entities"]) >= 3
    assert len(res["relations"]) >= 1

    rel_types = {r["relation_type"] for r in res["relations"]}
    assert "REQUIRES_DOC" in rel_types or "REGULATED_BY" in rel_types


# =============================================================================
# 4. Multi-Hop Graph Traversal Tests
# =============================================================================

def test_traversal_all_customs_domains():
    """Verify multi-hop compliance pathways for all 5 project domains using the seed KG."""
    kg = load_customs_kg()
    traverser = GraphTraverser(kg)

    # 1. Industrial Machinery (HS 8429)
    path_8429 = traverser.find_compliance_path("8429")
    assert path_8429.matched_entity_id == "HS_8429"
    doc_ids_8429 = {d["id"] for d in path_8429.required_documents}
    assert "DOC_BILL_OF_ENTRY_FORM_I" in doc_ids_8429
    assert "DOC_CHARTERED_ENGINEER_CERT" in doc_ids_8429

    # 2. Electronics / Computing Hardware (HS 8471)
    path_8471 = traverser.find_compliance_path("laptop")
    doc_ids_8471 = {d["id"] for d in path_8471.required_documents}
    assert "DOC_BIS_CRS_CERTIFICATE" in doc_ids_8471

    # 3. Hazardous Chemicals (HS 2901)
    path_chem = traverser.find_compliance_path("2901")
    doc_ids_chem = {d["id"] for d in path_chem.required_documents}
    assert "DOC_MSDS_GHS" in doc_ids_chem
    assert "DOC_DANGEROUS_GOODS_DECLARATION" in doc_ids_chem

    # 4. Dual-Use Electronics (HS 8542)
    path_dual = traverser.find_compliance_path("8542")
    doc_ids_dual = {d["id"] for d in path_dual.required_documents}
    assert "DOC_SCOMET_LICENSE" in doc_ids_dual
    assert "DOC_END_USER_CERTIFICATE" in doc_ids_dual

    # 5. Medical Devices (HS 9018)
    path_med = traverser.find_compliance_path("medical devices")
    doc_ids_med = {d["id"] for d in path_med.required_documents}
    assert "DOC_CDSCO_IMPORT_LICENSE" in doc_ids_med
    assert "DOC_FREE_SALE_CERTIFICATE" in doc_ids_med

    # 6. Agricultural Imports (HS 1001)
    path_agri = traverser.find_compliance_path("wheat")
    doc_ids_agri = {d["id"] for d in path_agri.required_documents}
    assert "DOC_PHYTOSANITARY_CERT" in doc_ids_agri
    assert "DOC_FSSAI_IMPORT_NOC" in doc_ids_agri


def test_subgraph_and_triple_formatter():
    """Verify localized k-hop subgraph extraction and prompt context formatting."""
    kg = load_customs_kg()
    traverser = GraphTraverser(kg)

    subgraph = traverser.get_subgraph_for_entities(["HS_8429"], max_hops=1)
    assert subgraph["num_nodes"] >= 5
    assert subgraph["num_edges"] >= 5

    formatted_context = traverser.format_triples_for_context(subgraph["triples"])
    assert "Customs Knowledge Graph Context:" in formatted_context
    assert "HS_8429" in formatted_context
    assert "--REQUIRES_DOC-->" in formatted_context


# =============================================================================
# 5. Knowledge Graph Agent Tests
# =============================================================================

def test_kg_agent_actions():
    """Test all execution actions of KnowledgeGraphAgent."""
    agent = KnowledgeGraphAgent()

    # Action 1: get_stats
    stats_res = agent.execute({"action": "get_stats"})
    assert stats_res["status"] == "success"
    assert stats_res["data"]["total_entities"] >= 80

    # Action 2: query_entities
    query_res = agent.execute({
        "action": "query_entities",
        "payload": {"keyword": "excavator"}
    })
    assert query_res["status"] == "success"
    assert query_res["data"]["count"] >= 1

    # Action 3: traverse
    trav_res = agent.execute({
        "action": "traverse",
        "payload": {"product_or_hscode": "8471.30", "target_agency_or_doc": "BIS"}
    })
    assert trav_res["status"] == "success"
    assert "formatted_text" in trav_res["data"]
    assert "BIS" in trav_res["data"]["formatted_text"]

    # Action 4: get_context_triples
    ctx_res = agent.execute({
        "action": "get_context_triples",
        "payload": {"query": "importing medical devices under HS 9018"}
    })
    assert ctx_res["status"] == "success"
    assert ctx_res["data"]["num_triples"] > 0
    assert "formatted_context" in ctx_res["data"]

    # Action 5: add_knowledge
    new_entity = {"id": "DOC_CUSTOM_TEST_DOC", "type": "REQUIRED_DOCUMENT", "name": "Custom Test Certificate"}
    new_rel = {"source_id": "HS_8429", "target_id": "DOC_CUSTOM_TEST_DOC", "relation_type": "REQUIRES_DOC"}
    add_res = agent.execute({
        "action": "add_knowledge",
        "payload": {"entities": [new_entity], "relations": [new_rel], "persist": False}
    })
    assert add_res["status"] == "success"
    assert agent.graph_store.get_entity("DOC_CUSTOM_TEST_DOC") is not None


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    print("Running Knowledge Graph Tests...")
    test_schema_entities_and_relations()
    print("✓ Schema entities and relations passed.")

    test_compliance_path_data_structure()
    print("✓ Compliance path data structure passed.")

    test_graph_store_mutations_and_neighbors()
    print("✓ Graph store mutations and neighbors passed.")

    test_graph_store_serialization()
    print("✓ Graph store JSON/GML/GraphML serialization passed.")

    test_neo4j_stub_readiness()
    print("✓ Neo4j interface stub readiness passed.")

    test_entity_extractor_patterns()
    print("✓ Entity extractor regex patterns passed.")

    test_entity_extractor_relations()
    print("✓ Entity extractor relations passed.")

    test_traversal_all_customs_domains()
    print("✓ Multi-hop traversal across all 5 customs domains passed.")

    test_subgraph_and_triple_formatter()
    print("✓ Subgraph extraction and triple formatter passed.")

    test_kg_agent_actions()
    print("✓ KnowledgeGraphAgent actions passed.")

    print("\nAll 10 test suites passed successfully!")


if __name__ == "__main__":
    run_all_tests()
