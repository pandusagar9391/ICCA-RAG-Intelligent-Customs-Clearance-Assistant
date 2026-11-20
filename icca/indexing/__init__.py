"""
Indexing module for ICCA-RAG.

Provides sparse inverted index (BM25) and dense vector storage (HNSW / FAISS / NumPy)
as defined in Section IV-B and Algorithm 2 of the ICCA-RAG paper.
"""

from icca.indexing.bm25_index import BM25Index
from icca.indexing.embeddings import EmbeddingsManager, DeterministicHashEmbedder
from icca.indexing.vector_index import VectorIndex

__all__ = [
    "BM25Index",
    "EmbeddingsManager",
    "DeterministicHashEmbedder",
    "VectorIndex",
]
