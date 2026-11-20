"""
Hybrid Retrieval Module for ICCA-RAG.

Combines Sparse Inverted Index (BM25) and Dense Vector Storage (FAISS/HNSW/NumPy)
as defined in Section IV-B.2 and IV-B.3 of the ICCA-RAG paper.

Features:
- Reciprocal Rank Fusion (RRF) with configurable smoothing constant k (default: 60).
- Weighted linear score combination with min-max normalization.
- Redundancy removal retaining the highest-scoring candidate among semantically duplicate text units.
- Comprehensive metadata tracking (dense scores, sparse scores, ranks, source documents).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from icca.config import default_config
from icca.indexing.bm25_index import BM25Index
from icca.indexing.embeddings import EmbeddingsManager
from icca.indexing.vector_index import VectorIndex

logger = logging.getLogger("HybridRetriever")


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate token-level Jaccard similarity for redundancy detection."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


class HybridRetriever:
    """
    Hybrid Retriever integrating BM25 sparse keyword retrieval with dense vector search.
    
    Attributes:
        bm25_index (BM25Index): Sparse inverted index instance.
        vector_index (VectorIndex): Dense vector storage instance.
        embeddings_manager (EmbeddingsManager): Embeddings generator instance.
        sparse_weight (float): Relative weight for BM25 results (default: 0.5).
        dense_weight (float): Relative weight for dense vector results (default: 0.5).
        rrf_k (int): Reciprocal Rank Fusion smoothing parameter (default: 60).
        redundancy_threshold (float): Jaccard threshold to filter redundant near-duplicate chunks.
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        vector_index: Optional[VectorIndex] = None,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        redundancy_threshold: float = 0.85,
    ):
        self.sparse_weight = float(sparse_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_k = int(rrf_k)
        self.redundancy_threshold = float(redundancy_threshold)

        # Initialize components if not provided
        self.bm25_index = bm25_index if bm25_index is not None else BM25Index(
            k1=default_config.bm25_k1,
            b=default_config.bm25_b,
        )
        self.vector_index = vector_index if vector_index is not None else VectorIndex(
            dimension=default_config.embedding_dimension,
        )
        self.embeddings_manager = embeddings_manager if embeddings_manager is not None else EmbeddingsManager(
            provider=getattr(default_config, "embedding_provider", "hash"),
            model_name=default_config.dense_model_name,
            dimension=default_config.embedding_dimension,
        )

    def index_documents(self, documents: List[Union[Dict[str, Any], str]]) -> int:
        """
        Simultaneously index a collection of document chunks into both sparse and dense indices.
        
        Args:
            documents: List of chunk dictionaries or raw strings.
                       Dict format: {'id': '...', 'text': '...', 'metadata': {...}}
                       
        Returns:
            Number of indexed chunks.
        """
        if not documents:
            return 0

        # Standardize chunks
        formatted_chunks: List[Dict[str, Any]] = []
        texts: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, item in enumerate(documents):
            if isinstance(item, dict):
                chunk_id = str(item.get("id") or item.get("chunk_id") or f"chunk_{i}")
                text = str(item.get("text") or item.get("content") or "")
                meta = dict(item.get("metadata", {}))
                for k, v in item.items():
                    if k not in ("id", "chunk_id", "text", "content", "metadata"):
                        meta[k] = v
            else:
                chunk_id = f"chunk_{i}"
                text = str(item)
                meta = {}

            meta["id"] = chunk_id
            meta["text"] = text

            formatted_chunks.append({"id": chunk_id, "text": text, "metadata": meta})
            texts.append(text)
            ids.append(chunk_id)
            metadatas.append(meta)

        # 1. Index into BM25 Sparse Inverted Index
        logger.info(f"Indexing {len(formatted_chunks)} chunks into BM25 inverted index...")
        self.bm25_index.index_chunks(formatted_chunks)

        # 2. Compute Dense Embeddings & Index into Vector Store
        logger.info(f"Generating dense embeddings for {len(texts)} chunks via {self.embeddings_manager.backend_name}...")
        vectors = self.embeddings_manager.get_embeddings(texts)
        self.vector_index.add_vectors(vectors, ids, metadatas)

        logger.info(f"Successfully indexed {len(ids)} chunks in both sparse and dense stores.")
        return len(ids)

    def retrieve_sparse(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform BM25 sparse keyword search."""
        return self.bm25_index.search(query, top_k=top_k)

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform dense semantic vector search."""
        query_vector = self.embeddings_manager.get_embedding(query)
        return self.vector_index.search(query_vector, top_k=top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "rrf",
        candidate_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval combining sparse keyword and dense semantic results.
        
        Args:
            query: User search query.
            top_k: Number of final results to return.
            mode: Fusion strategy: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (Linear score fusion).
            candidate_k: Number of intermediate candidates to retrieve from each index (default: top_k * 2).
            
        Returns:
            List of ranked result dictionaries with fused scores and individual modality scores.
        """
        if not query.strip():
            return []

        cand_k = candidate_k or max(top_k * 2, 10)

        # 1. Retrieve top candidates from both channels
        sparse_results = self.retrieve_sparse(query, top_k=cand_k)
        dense_results = self.retrieve_dense(query, top_k=cand_k)

        if not sparse_results and not dense_results:
            return []

        # 2. Perform score fusion
        if mode.lower() == "weighted":
            fused_candidates = self._fuse_weighted(sparse_results, dense_results)
        else:
            fused_candidates = self._fuse_rrf(sparse_results, dense_results)

        # 3. Apply Redundancy Removal (Section IV-B.3)
        deduplicated = self._remove_redundancy(fused_candidates)

        # 4. Take top_k and re-rank
        final_results = deduplicated[:top_k]
        for rank, res in enumerate(final_results, start=1):
            res["final_rank"] = rank

        return final_results

    def _fuse_rrf(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
            Score(d) = sum_{m in {sparse, dense}} w_m / (k + rank_m(d))
        """
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process Sparse candidates
        for item in sparse_results:
            doc_id = item["id"]
            rank = item["rank"]
            score = item["score"]
            rrf_val = self.sparse_weight / float(self.rrf_k + rank)

            doc_map[doc_id] = {
                "id": doc_id,
                "score": rrf_val,
                "rrf_score": rrf_val,
                "sparse_score": score,
                "sparse_rank": rank,
                "dense_score": None,
                "dense_rank": None,
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
            }

        # Process Dense candidates
        for item in dense_results:
            doc_id = item["id"]
            rank = item["rank"]
            score = item["score"]
            rrf_val = self.dense_weight / float(self.rrf_k + rank)

            if doc_id in doc_map:
                doc_map[doc_id]["score"] += rrf_val
                doc_map[doc_id]["rrf_score"] += rrf_val
                doc_map[doc_id]["dense_score"] = score
                doc_map[doc_id]["dense_rank"] = rank
                # Prefer existing text or fallback
                if not doc_map[doc_id]["text"]:
                    doc_map[doc_id]["text"] = item.get("text", "")
            else:
                doc_map[doc_id] = {
                    "id": doc_id,
                    "score": rrf_val,
                    "rrf_score": rrf_val,
                    "sparse_score": None,
                    "sparse_rank": None,
                    "dense_score": score,
                    "dense_rank": rank,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                }

        # Sort descending by fused score
        sorted_candidates = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)
        for cand in sorted_candidates:
            cand["score"] = round(cand["score"], 6)

        return sorted_candidates

    def _fuse_weighted(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Weighted linear score combination with min-max normalization:
            Score(d) = w_sparse * norm(sparse_score) + w_dense * norm(dense_score)
        """
        # Min-max normalize sparse scores
        sparse_scores = [r["score"] for r in sparse_results]
        s_min = min(sparse_scores) if sparse_scores else 0.0
        s_max = max(sparse_scores) if sparse_scores else 1.0
        s_has_range = (s_max - s_min) > 1e-6

        # Min-max normalize dense scores
        dense_scores = [r["score"] for r in dense_results]
        d_min = min(dense_scores) if dense_scores else 0.0
        d_max = max(dense_scores) if dense_scores else 1.0
        d_has_range = (d_max - d_min) > 1e-6

        doc_map: Dict[str, Dict[str, Any]] = {}

        for item in sparse_results:
            doc_id = item["id"]
            raw_s = item["score"]
            norm_s = ((raw_s - s_min) / (s_max - s_min)) if s_has_range else (1.0 if raw_s > 0 else 0.0)
            fused = self.sparse_weight * norm_s

            doc_map[doc_id] = {
                "id": doc_id,
                "score": fused,
                "sparse_score": raw_s,
                "sparse_norm": round(norm_s, 4),
                "sparse_rank": item["rank"],
                "dense_score": None,
                "dense_norm": 0.0,
                "dense_rank": None,
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
            }

        for item in dense_results:
            doc_id = item["id"]
            raw_d = item["score"]
            norm_d = ((raw_d - d_min) / (d_max - d_min)) if d_has_range else (1.0 if raw_d > 0 else 0.0)
            dense_contrib = self.dense_weight * norm_d

            if doc_id in doc_map:
                doc_map[doc_id]["score"] += dense_contrib
                doc_map[doc_id]["dense_score"] = raw_d
                doc_map[doc_id]["dense_norm"] = round(norm_d, 4)
                doc_map[doc_id]["dense_rank"] = item["rank"]
                if not doc_map[doc_id]["text"]:
                    doc_map[doc_id]["text"] = item.get("text", "")
            else:
                doc_map[doc_id] = {
                    "id": doc_id,
                    "score": dense_contrib,
                    "sparse_score": None,
                    "sparse_norm": 0.0,
                    "sparse_rank": None,
                    "dense_score": raw_d,
                    "dense_norm": round(norm_d, 4),
                    "dense_rank": item["rank"],
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                }

        sorted_candidates = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)
        for cand in sorted_candidates:
            cand["score"] = round(cand["score"], 4)

        return sorted_candidates

    def _remove_redundancy(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Redundancy removal strategy (Section IV-B.3):
        Filters out near-duplicate chunks based on token overlap (Jaccard similarity),
        retaining only the higher-scoring chunk.
        """
        retained: List[Dict[str, Any]] = []

        for candidate in candidates:
            text = candidate.get("text", "").strip()
            if not text:
                retained.append(candidate)
                continue

            # Compare with already retained candidates
            is_redundant = False
            for accepted in retained:
                accepted_text = accepted.get("text", "").strip()
                if not accepted_text:
                    continue

                sim = compute_jaccard_similarity(text, accepted_text)
                if sim >= self.redundancy_threshold:
                    # Near-duplicate detected; skip this lower-scoring candidate
                    is_redundant = True
                    logger.debug(
                        f"Filtered redundant candidate {candidate['id']} "
                        f"(similar to {accepted['id']} with Jaccard={sim:.2f})"
                    )
                    break

            if not is_redundant:
                retained.append(candidate)

        return retained

    def save(self, storage_dir: Union[str, Path]) -> None:
        """
        Persist both BM25 and Vector indices into a directory.
        
        Args:
            storage_dir: Path to directory where index files will be stored.
        """
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        bm25_file = storage_path / "bm25_index.json"
        vector_file = storage_path / "vector_index.json"

        logger.info(f"Saving BM25 index to: {bm25_file}")
        self.bm25_index.save(bm25_file)

        logger.info(f"Saving Vector index to: {vector_file}")
        self.vector_index.save(vector_file)

    def load(self, storage_dir: Union[str, Path]) -> None:
        """
        Load both BM25 and Vector indices from a directory.
        
        Args:
            storage_dir: Path to directory containing saved index files.
        """
        storage_path = Path(storage_dir)
        bm25_file = storage_path / "bm25_index.json"
        vector_file = storage_path / "vector_index.json"

        if bm25_file.exists():
            logger.info(f"Loading BM25 index from: {bm25_file}")
            self.bm25_index.load(bm25_file)
        else:
            logger.warning(f"BM25 index file not found at: {bm25_file}")

        if vector_file.exists():
            logger.info(f"Loading Vector index from: {vector_file}")
            self.vector_index.load(vector_file)
        else:
            logger.warning(f"Vector index file not found at: {vector_file}")
