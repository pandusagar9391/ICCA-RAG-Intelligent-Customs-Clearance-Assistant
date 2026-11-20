"""
Retrieval module for ICCA-RAG.

Provides hybrid retrieval combining sparse BM25 and dense vector search
with Reciprocal Rank Fusion (RRF), weighted fusion, and redundancy removal.
"""

from icca.retrieval.hybrid_retriever import HybridRetriever, compute_jaccard_similarity
from icca.retrieval.query_reformulator import QueryReformulator
from icca.retrieval.context_builder import ContextBuilder

__all__ = [
    "HybridRetriever",
    "compute_jaccard_similarity",
    "QueryReformulator",
    "ContextBuilder",
]
