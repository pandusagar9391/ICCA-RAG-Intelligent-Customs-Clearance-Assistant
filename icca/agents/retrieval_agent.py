"""
Retrieval Agent for ICCA-RAG.

Specialized agent responsible for coordinating sparse (BM25), dense (vector),
and hybrid multi-modal retrieval operations as defined in Section IV-B of the paper.
Inherits from BaseAgent.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from icca.agents.base_agent import BaseAgent
from icca.config import ICCAConfig, default_config
from icca.indexing.bm25_index import BM25Index
from icca.indexing.embeddings import EmbeddingsManager
from icca.indexing.vector_index import VectorIndex
from icca.retrieval.hybrid_retriever import HybridRetriever


class RetrievalAgent(BaseAgent):
    """
    Retrieval Specialist Agent for ICCA-RAG.
    
    Coordinates the sparse inverted index (BM25) and dense vector store (HNSW/FAISS/NumPy),
    executing individual or fused hybrid search tasks.
    
    Supported Actions:
        - 'index': Index incoming text chunks into sparse and dense stores.
        - 'search_sparse': Perform BM25 keyword retrieval.
        - 'search_dense': Perform dense semantic vector retrieval.
        - 'search_hybrid': Perform hybrid retrieval with RRF or weighted fusion.
        - 'save': Persist indexes to disk.
        - 'load': Load indexes from disk.
    """

    def __init__(
        self,
        name: str = "retrieval_agent",
        role: str = "Retrieval Specialist",
        config: Optional[ICCAConfig] = None,
    ):
        super().__init__(name=name, role=role)
        self.config = config or default_config

        # Initialize indexing components
        self.bm25_index = BM25Index(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        self.embeddings_manager = EmbeddingsManager(
            provider=getattr(self.config, "embedding_provider", "hash"),
            model_name=self.config.dense_model_name,
            dimension=self.config.embedding_dimension,
        )
        self.vector_index = VectorIndex(
            dimension=self.config.embedding_dimension,
        )

        # Initialize hybrid retrieval coordinator
        self.retriever = HybridRetriever(
            bm25_index=self.bm25_index,
            vector_index=self.vector_index,
            embeddings_manager=self.embeddings_manager,
        )

        self.log(
            f"RetrievalAgent initialized with {self.embeddings_manager.backend_name} embeddings "
            f"and {self.vector_index.backend} vector store."
        )

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute delegated retrieval task and return standardized response dictionary.
        
        Args:
            task: Dict containing 'action' and 'payload'.
            
        Returns:
            Dict containing 'status' ('success' or 'error'), 'data', and 'message'.
        """
        if not isinstance(task, dict):
            return {
                "status": "error",
                "data": None,
                "message": f"Task must be a dictionary, got {type(task).__name__}.",
            }

        action = task.get("action", "").lower().strip()
        payload = task.get("payload", {})
        if isinstance(payload, dict):
            # Combine task-level keys with payload so both formats are supported
            effective_payload = {**task, **payload}
        else:
            effective_payload = {**task, "query": str(payload)}
        payload = effective_payload

        try:
            if action == "index":
                return self._handle_index(payload)
            elif action == "search_sparse":
                return self._handle_search_sparse(payload)
            elif action == "search_dense":
                return self._handle_search_dense(payload)
            elif action == "search_hybrid":
                return self._handle_search_hybrid(payload)
            elif action == "save":
                return self._handle_save(payload)
            elif action == "load":
                return self._handle_load(payload)
            else:
                return {
                    "status": "error",
                    "data": None,
                    "message": (
                        f"Unrecognized retrieval action: '{action}'. Supported actions: "
                        f"'index', 'search_sparse', 'search_dense', 'search_hybrid', 'save', 'load'."
                    ),
                }

        except Exception as e:
            self.log(f"Error during execution of action '{action}': {str(e)}", level=logging.ERROR)
            return {
                "status": "error",
                "data": None,
                "message": f"Execution error in {action}: {str(e)}",
            }

    def _handle_index(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Index chunks into both sparse and dense stores."""
        chunks = payload.get("chunks") or payload.get("documents") or []
        if not chunks:
            return {
                "status": "error",
                "data": None,
                "message": "Payload for 'index' action must provide non-empty 'chunks' or 'documents'.",
            }

        count = self.retriever.index_documents(chunks)
        self.log(f"Indexed {count} chunks into BM25 and Vector stores.")

        return {
            "status": "success",
            "data": {
                "indexed_count": count,
                "bm25_total_docs": self.bm25_index.total_docs,
                "vector_total_docs": len(self.vector_index),
            },
            "message": f"Successfully indexed {count} document chunks.",
        }

    def _handle_search_sparse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle BM25 sparse search."""
        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", self.config.sparse_top_k))

        if not query:
            return {"status": "error", "data": None, "message": "Search query cannot be empty."}

        results = self.retriever.retrieve_sparse(query=query, top_k=top_k)
        self.log(f"Sparse search for '{query}' returned {len(results)} results.")

        return {
            "status": "success",
            "data": {
                "query": query,
                "results": results,
                "sparse": results,
                "count": len(results),
            },
            "message": f"Retrieved {len(results)} sparse results.",
        }

    def _handle_search_dense(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dense vector search."""
        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", self.config.dense_top_k))

        if not query:
            return {"status": "error", "data": None, "message": "Search query cannot be empty."}

        results = self.retriever.retrieve_dense(query=query, top_k=top_k)
        self.log(f"Dense search for '{query}' returned {len(results)} results.")

        return {
            "status": "success",
            "data": {
                "query": query,
                "results": results,
                "dense": results,
                "count": len(results),
            },
            "message": f"Retrieved {len(results)} dense results.",
        }

    def _handle_search_hybrid(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle full hybrid retrieval."""
        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", 5))
        mode = str(payload.get("mode", "rrf")).lower()

        if not query:
            return {"status": "error", "data": None, "message": "Search query cannot be empty."}

        results = self.retriever.retrieve(query=query, top_k=top_k, mode=mode)
        sparse_results = self.retriever.retrieve_sparse(query=query, top_k=top_k)
        dense_results = self.retriever.retrieve_dense(query=query, top_k=top_k)
        self.log(f"Hybrid ({mode.upper()}) search for '{query}' returned {len(results)} results.")

        return {
            "status": "success",
            "data": {
                "query": query,
                "mode": mode,
                "results": results,
                "hybrid": results,
                "sparse": sparse_results,
                "dense": dense_results,
                "count": len(results),
            },
            "message": f"Retrieved {len(results)} hybrid results using {mode.upper()} fusion.",
        }

    def _handle_save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Persist indices to disk."""
        target_dir = payload.get("path") or str(self.config.storage_dir)
        self.retriever.save(target_dir)
        self.log(f"Persisted indexes to directory: {target_dir}")

        return {
            "status": "success",
            "data": {"storage_dir": target_dir},
            "message": f"Indexes successfully saved to {target_dir}.",
        }

    def _handle_load(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Load indices from disk."""
        target_dir = payload.get("path") or str(self.config.storage_dir)
        self.retriever.load(target_dir)
        self.log(f"Loaded indexes from directory: {target_dir}")

        return {
            "status": "success",
            "data": {
                "storage_dir": target_dir,
                "bm25_count": self.bm25_index.total_docs,
                "vector_count": len(self.vector_index),
            },
            "message": (
                f"Indexes successfully loaded from {target_dir} "
                f"({self.bm25_index.total_docs} BM25 docs, {len(self.vector_index)} vectors)."
            ),
        }
