"""
ICCA-RAG Multi-Agent Module.
"""

from icca.agents.base_agent import BaseAgent
from icca.agents.ingestion_agent import DataIngestionAgent
from icca.agents.compliance_agent import ComplianceAgent
from icca.agents.kg_agent import KnowledgeGraphAgent
from icca.agents.retrieval_agent import RetrievalAgent

__all__ = [
    "BaseAgent",
    "DataIngestionAgent",
    "ComplianceAgent",
    "KnowledgeGraphAgent",
    "RetrievalAgent",
]
