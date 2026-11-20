"""
Knowledge Graph Package for ICCA-RAG.

Provides:
- Schema definitions (CustomsEntity, CustomsRelation, CompliancePath, EntityType, RelationType)
- Graph storage and serialization (CustomsGraphStore, NetworkX / InMemory DiGraph)
- Entity & relation extraction (CustomsEntityExtractor)
- Multi-hop traversal and prompt formatting (GraphTraverser)
- Helper functions to load the default customs knowledge graph
"""

from pathlib import Path
from typing import Optional, Union

from icca.knowledge_graph.schema import (
    CompliancePath,
    CustomsEntity,
    CustomsRelation,
    EntityType,
    RelationType,
    HS_CODE,
    PRODUCT_CATEGORY,
    REGULATION,
    REQUIRED_DOCUMENT,
    TARIFF_RATE,
    COUNTRY,
    INSPECTION_RULE,
    GOVERNMENT_AGENCY,
    REQUIRES_DOC,
    REGULATED_BY,
    HAS_TARIFF,
    CLASSIFIED_AS,
    SUBJECT_TO,
    EXEMPT_UNDER,
    ISSUED_BY,
)
from icca.knowledge_graph.graph_store import CustomsGraphStore
from icca.knowledge_graph.entity_extractor import CustomsEntityExtractor
from icca.knowledge_graph.traverser import GraphTraverser
from icca.config import default_config


def load_customs_kg(kg_path: Optional[Union[str, Path]] = None) -> CustomsGraphStore:
    """
    Factory helper to load the default pre-populated customs knowledge graph.
    
    Args:
        kg_path: Optional explicit path to customs_kg.json.
                 Defaults to data/knowledge_graph/customs_kg.json in project config.
                 
    Returns:
        Populated CustomsGraphStore instance.
    """
    path = Path(kg_path) if kg_path else default_config.kg_dir / "customs_kg.json"
    store = CustomsGraphStore(name="ICCA_Customs_KG")
    if path.exists():
        store.load_json(path)
    return store


__all__ = [
    "CustomsGraphStore",
    "CustomsEntityExtractor",
    "GraphTraverser",
    "CustomsEntity",
    "CustomsRelation",
    "CompliancePath",
    "EntityType",
    "RelationType",
    "load_customs_kg",
    "HS_CODE",
    "PRODUCT_CATEGORY",
    "REGULATION",
    "REQUIRED_DOCUMENT",
    "TARIFF_RATE",
    "COUNTRY",
    "INSPECTION_RULE",
    "GOVERNMENT_AGENCY",
    "REQUIRES_DOC",
    "REGULATED_BY",
    "HAS_TARIFF",
    "CLASSIFIED_AS",
    "SUBJECT_TO",
    "EXEMPT_UNDER",
    "ISSUED_BY",
]
