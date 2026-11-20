"""
Customs Knowledge Graph Schema Definitions for ICCA-RAG.

Defines the core ontological schema:
- Entity Types:
    HS_CODE, PRODUCT_CATEGORY, REGULATION, REQUIRED_DOCUMENT,
    TARIFF_RATE, COUNTRY, INSPECTION_RULE, GOVERNMENT_AGENCY.
- Relation Types:
    REQUIRES_DOC, REGULATED_BY, HAS_TARIFF, CLASSIFIED_AS,
    SUBJECT_TO, EXEMPT_UNDER, ISSUED_BY.
- Data Structures:
    CustomsEntity, CustomsRelation, CompliancePath.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class EntityType(str, Enum):
    """Statutory and trade entity types recognized in ICCA-RAG."""
    HS_CODE = "HS_CODE"
    PRODUCT_CATEGORY = "PRODUCT_CATEGORY"
    REGULATION = "REGULATION"
    REQUIRED_DOCUMENT = "REQUIRED_DOCUMENT"
    TARIFF_RATE = "TARIFF_RATE"
    COUNTRY = "COUNTRY"
    INSPECTION_RULE = "INSPECTION_RULE"
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"


class RelationType(str, Enum):
    """Semantic relation types connecting customs entities."""
    REQUIRES_DOC = "REQUIRES_DOC"
    REGULATED_BY = "REGULATED_BY"
    HAS_TARIFF = "HAS_TARIFF"
    CLASSIFIED_AS = "CLASSIFIED_AS"
    SUBJECT_TO = "SUBJECT_TO"
    EXEMPT_UNDER = "EXEMPT_UNDER"
    ISSUED_BY = "ISSUED_BY"


# Top-level string constants for convenience
HS_CODE = EntityType.HS_CODE.value
PRODUCT_CATEGORY = EntityType.PRODUCT_CATEGORY.value
REGULATION = EntityType.REGULATION.value
REQUIRED_DOCUMENT = EntityType.REQUIRED_DOCUMENT.value
TARIFF_RATE = EntityType.TARIFF_RATE.value
COUNTRY = EntityType.COUNTRY.value
INSPECTION_RULE = EntityType.INSPECTION_RULE.value
GOVERNMENT_AGENCY = EntityType.GOVERNMENT_AGENCY.value

REQUIRES_DOC = RelationType.REQUIRES_DOC.value
REGULATED_BY = RelationType.REGULATED_BY.value
HAS_TARIFF = RelationType.HAS_TARIFF.value
CLASSIFIED_AS = RelationType.CLASSIFIED_AS.value
SUBJECT_TO = RelationType.SUBJECT_TO.value
EXEMPT_UNDER = RelationType.EXEMPT_UNDER.value
ISSUED_BY = RelationType.ISSUED_BY.value


@dataclass
class CustomsEntity:
    """
    Node in the Customs Knowledge Graph representing an entity such as
    an HS Code, statutory document, regulatory body, or inspection mandate.
    """
    id: str
    type: Union[str, EntityType]
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.type, EntityType):
            self.type = self.type.value
        self.id = str(self.id).strip()
        self.name = str(self.name).strip()
        if not isinstance(self.properties, dict):
            self.properties = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to a serializable dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomsEntity":
        """Reconstruct entity from dictionary representation."""
        return cls(
            id=str(data["id"]),
            type=data.get("type", "UNKNOWN"),
            name=str(data.get("name", data["id"])),
            description=str(data.get("description", "")),
            properties=dict(data.get("properties", {})),
        )


@dataclass
class CustomsRelation:
    """
    Directed semantic edge between two CustomsEntity nodes in the knowledge graph.
    """
    source_id: str
    target_id: str
    relation_type: Union[str, RelationType]
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self):
        if isinstance(self.relation_type, RelationType):
            self.relation_type = self.relation_type.value
        self.source_id = str(self.source_id).strip()
        self.target_id = str(self.target_id).strip()
        self.weight = float(self.weight)
        if not isinstance(self.properties, dict):
            self.properties = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to a serializable dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": dict(self.properties),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomsRelation":
        """Reconstruct relation from dictionary representation."""
        return cls(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            relation_type=data.get("relation_type", data.get("type", "RELATED_TO")),
            properties=dict(data.get("properties", {})),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class CompliancePath:
    """
    Structured result of multi-hop compliance path traversal from a product
    or HS code to its applicable regulations, required certificates, agencies,
    and inspection rules.
    """
    product_or_hscode: str
    matched_entity_id: Optional[str] = None
    hs_codes: List[Dict[str, Any]] = field(default_factory=list)
    regulations: List[Dict[str, Any]] = field(default_factory=list)
    required_documents: List[Dict[str, Any]] = field(default_factory=list)
    agencies: List[Dict[str, Any]] = field(default_factory=list)
    inspection_rules: List[Dict[str, Any]] = field(default_factory=list)
    tariff_rates: List[Dict[str, Any]] = field(default_factory=list)
    triples: List[Tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert compliance path into a serializable dictionary."""
        return {
            "product_or_hscode": self.product_or_hscode,
            "matched_entity_id": self.matched_entity_id,
            "hs_codes": self.hs_codes,
            "regulations": self.regulations,
            "required_documents": self.required_documents,
            "agencies": self.agencies,
            "inspection_rules": self.inspection_rules,
            "tariff_rates": self.tariff_rates,
            "triples": self.triples,
            "summary": self.summary,
        }

    def format_text(self) -> str:
        """Render a formatted, human-readable compliance overview."""
        lines = [
            f"=== ICCA Compliance Clearance Pathway for '{self.product_or_hscode}' ===",
            f"Primary Node: {self.matched_entity_id or 'N/A'}",
            "",
            "1. Harmonized System Classification(s):",
        ]
        if self.hs_codes:
            for item in self.hs_codes:
                lines.append(f"   * [{item.get('id')}] {item.get('name')}: {item.get('description', '')}")
        else:
            lines.append("   * None identified")

        lines.append("\n2. Applicable Statutory Regulations:")
        if self.regulations:
            for item in self.regulations:
                lines.append(f"   * [{item.get('id')}] {item.get('name')}: {item.get('description', '')}")
        else:
            lines.append("   * None identified")

        lines.append("\n3. Mandatory Clearance Documents & Certificates:")
        if self.required_documents:
            for item in self.required_documents:
                props = item.get("properties", {})
                mandatory_str = " (Mandatory)" if props.get("mandatory", True) else " (Conditional)"
                lines.append(f"   * [{item.get('id')}] {item.get('name')}{mandatory_str}: {item.get('description', '')}")
        else:
            lines.append("   * None identified")

        lines.append("\n4. Competent Regulatory Authorities & Agencies:")
        if self.agencies:
            for item in self.agencies:
                lines.append(f"   * [{item.get('id')}] {item.get('name')}: {item.get('description', '')}")
        else:
            lines.append("   * None identified")

        lines.append("\n5. Applicable Inspection & Port Clearance Rules:")
        if self.inspection_rules:
            for item in self.inspection_rules:
                lines.append(f"   * [{item.get('id')}] {item.get('name')}: {item.get('description', '')}")
        else:
            lines.append("   * Standard RMS customs appraisal")

        lines.append("\n6. Tariff & Duty Schedule:")
        if self.tariff_rates:
            for item in self.tariff_rates:
                props = item.get("properties", {})
                bcd = props.get("basic_customs_duty", "N/A")
                igst = props.get("igst", "N/A")
                sws = props.get("social_welfare_surcharge", "10%")
                lines.append(f"   * [{item.get('id')}] {item.get('name')}: BCD={bcd}, IGST={igst}, SWS={sws}")
        else:
            lines.append("   * Tariff rate not explicitly mapped")

        if self.summary:
            lines.append(f"\nSummary:\n{self.summary}")

        return "\n".join(lines)
