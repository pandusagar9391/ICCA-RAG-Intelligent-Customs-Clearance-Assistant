"""
Dense Vector Storage and Approximate Nearest Neighbor (ANN) Index for ICCA-RAG.

Implements dense retrieval as specified in Section IV-B, Algorithm 2 (HNSW),
and Equation (6) of the ICCA-RAG paper:
    Similarity(vq, vi) = (vq . vi) / (||vq|| ||vi||)

Supports:
1. FAISS HNSW / Flat Inner Product index if `faiss` is installed.
2. Fast NumPy matrix-based Cosine / Dot-product similarity search.
3. Robust pure-Python Cosine similarity fallback if neither FAISS nor NumPy is present.
4. Identical API regardless of underlying backend:
   - add_vectors(vectors, ids, metadatas)
   - search(query_vector, top_k)
   - save(path)
   - load(path)
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("VectorIndex")

# Check for NumPy
try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore

# Check for FAISS
try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    faiss = None  # type: ignore


def normalize_vector(v: Sequence[float]) -> List[float]:
    """L2-normalize a float vector to unit length."""
    norm_sq = sum(x * x for x in v)
    if norm_sq > 0.0:
        inv_norm = 1.0 / math.sqrt(norm_sq)
        return [float(x * inv_norm) for x in v]
    return [float(x) for x in v]


class VectorIndex:
    """
    Dense Vector Index supporting HNSW, FAISS, NumPy, and pure-Python fallbacks.
    
    Attributes:
        dimension (int): Dimensionality of dense vectors (default: 384).
        use_hnsw (bool): Whether to build HNSW graph when FAISS is available.
        backend (str): 'faiss_hnsw', 'faiss_flat', 'numpy', or 'python'.
    """

    def __init__(
        self,
        dimension: int = 384,
        use_hnsw: bool = True,
        metric: str = "cosine",
    ):
        self.dimension = int(dimension)
        self.use_hnsw = use_hnsw
        self.metric = metric.lower()

        # In-memory storage
        self.ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vectors: List[List[float]] = []
        self.id_to_idx: Dict[str, int] = {}

        # FAISS index handle
        self._faiss_index = None
        self.backend = self._determine_backend()

    def _determine_backend(self) -> str:
        """Select appropriate backend based on installed libraries."""
        if _HAS_FAISS:
            if self.use_hnsw:
                try:
                    # M=32 links per node in HNSW graph (Algorithm 2)
                    self._faiss_index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
                    return "faiss_hnsw"
                except Exception as e:
                    logger.warning(f"Failed to create FAISS HNSW index: {e}. Trying FlatIP.")
            try:
                self._faiss_index = faiss.IndexFlatIP(self.dimension)
                return "faiss_flat"
            except Exception as e:
                logger.warning(f"Failed to create FAISS FlatIP index: {e}. Falling back to NumPy.")

        if _HAS_NUMPY:
            return "numpy"

        return "python"

    def add_vectors(
        self,
        vectors: Union[List[List[float]], Any],
        ids: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add batch of dense vectors and associated identifiers into the index.
        
        Args:
            vectors: 2D array or list of float vectors (N x dimension).
            ids: List of unique document identifiers (length N).
            metadatas: Optional list of metadata dictionaries (length N).
        """
        n = len(ids)
        if n == 0:
            return

        # Convert vectors to normalized Python list of lists
        norm_vectors: List[List[float]] = []
        for i in range(n):
            vec = vectors[i]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            # If metric is cosine, normalize to unit length for inner product equivalence
            if self.metric == "cosine":
                vec = normalize_vector(vec)
            else:
                vec = [float(x) for x in vec]
            norm_vectors.append(vec)

        # Handle FAISS if active
        if self._faiss_index is not None and _HAS_FAISS and _HAS_NUMPY:
            np_arr = np.array(norm_vectors, dtype=np.float32)
            self._faiss_index.add(np_arr)

        # Store in internal registry
        for i, doc_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            meta["id"] = str(doc_id)

            if doc_id in self.id_to_idx:
                # Update existing vector
                existing_idx = self.id_to_idx[doc_id]
                self.vectors[existing_idx] = norm_vectors[i]
                self.metadatas[existing_idx] = meta
            else:
                idx = len(self.ids)
                self.ids.append(str(doc_id))
                self.vectors.append(norm_vectors[i])
                self.metadatas.append(meta)
                self.id_to_idx[str(doc_id)] = idx

    def search(
        self,
        query_vector: Union[List[float], Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search nearest neighbors using Cosine / Inner Product similarity.
        
        Args:
            query_vector: Dense query vector.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of dicts: [{'id': ..., 'score': ..., 'rank': ..., 'metadata': ...}]
            sorted by similarity score descending.
        """
        total = len(self.ids)
        if total == 0:
            return []

        top_k = min(top_k, total)

        # Convert to list and normalize
        if hasattr(query_vector, "tolist"):
            q_vec = query_vector.tolist()
        else:
            q_vec = list(query_vector)

        if self.metric == "cosine":
            q_vec = normalize_vector(q_vec)

        # Strategy 1: FAISS Search
        if self._faiss_index is not None and _HAS_FAISS and _HAS_NUMPY and self._faiss_index.ntotal == total:
            try:
                q_arr = np.array([q_vec], dtype=np.float32)
                distances, indices = self._faiss_index.search(q_arr, top_k)
                results: List[Dict[str, Any]] = []
                for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
                    if idx < 0 or idx >= total:
                        continue
                    doc_id = self.ids[idx]
                    meta = self.metadatas[idx]
                    results.append({
                        "id": doc_id,
                        "score": round(float(score), 4),
                        "rank": rank,
                        "text": meta.get("text", ""),
                        "metadata": {k: v for k, v in meta.items() if k not in ("text",)},
                    })
                return results
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}. Falling back to NumPy/Python search.")

        # Strategy 2: NumPy Cosine / Dot Product Matrix Multiply
        if _HAS_NUMPY:
            try:
                matrix = np.array(self.vectors, dtype=np.float32)
                q_arr = np.array(q_vec, dtype=np.float32)
                scores = np.dot(matrix, q_arr)
                # Get top_k indices
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = []
                for rank, idx in enumerate(top_indices, start=1):
                    doc_id = self.ids[idx]
                    meta = self.metadatas[idx]
                    results.append({
                        "id": doc_id,
                        "score": round(float(scores[idx]), 4),
                        "rank": rank,
                        "text": meta.get("text", ""),
                        "metadata": {k: v for k, v in meta.items() if k not in ("text",)},
                    })
                return results
            except Exception as e:
                logger.warning(f"NumPy vector search failed: {e}. Falling back to pure Python.")

        # Strategy 3: Pure-Python Cosine / Dot Product
        doc_scores = []
        for idx, stored_vec in enumerate(self.vectors):
            score = sum(a * b for a, b in zip(q_vec, stored_vec))
            doc_scores.append((idx, score))

        # Sort descending
        doc_scores.sort(key=lambda item: item[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(doc_scores[:top_k], start=1):
            doc_id = self.ids[idx]
            meta = self.metadatas[idx]
            results.append({
                "id": doc_id,
                "score": round(float(score), 4),
                "rank": rank,
                "text": meta.get("text", ""),
                "metadata": {k: v for k, v in meta.items() if k not in ("text",)},
            })

        return results

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize vectors and metadata to disk.
        
        Args:
            path: Destination file path (e.g., 'vectors.json').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "dimension": self.dimension,
            "metric": self.metric,
            "backend": self.backend,
            "ids": self.ids,
            "metadatas": self.metadatas,
            "vectors": self.vectors,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load vectors and metadata from disk and reconstruct index.
        
        Args:
            path: Source file path.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vector index file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.dimension = int(payload.get("dimension", 384))
        self.metric = payload.get("metric", "cosine")
        self.ids = payload.get("ids", [])
        self.metadatas = payload.get("metadatas", [])
        self.vectors = payload.get("vectors", [])
        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}

        # Re-initialize backend & FAISS if applicable
        self.backend = self._determine_backend()
        if self._faiss_index is not None and _HAS_FAISS and _HAS_NUMPY and len(self.vectors) > 0:
            np_arr = np.array(self.vectors, dtype=np.float32)
            self._faiss_index.add(np_arr)

    def clear(self) -> None:
        """Reset the vector index to empty."""
        self.ids.clear()
        self.metadatas.clear()
        self.vectors.clear()
        self.id_to_idx.clear()
        self._faiss_index = None
        self.backend = self._determine_backend()

    def __len__(self) -> int:
        return len(self.ids)
