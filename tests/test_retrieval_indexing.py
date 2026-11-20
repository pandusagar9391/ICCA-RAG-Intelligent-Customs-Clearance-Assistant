"""
Unit tests for ICCA-RAG Indexing and Retrieval components.
Tests BM25 inverted index, embeddings manager, vector index, hybrid retriever, and retrieval agent.
"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from icca.indexing.bm25_index import BM25Index, simple_porter_stem
from icca.indexing.embeddings import EmbeddingsManager, DeterministicHashEmbedder
from icca.indexing.vector_index import VectorIndex, normalize_vector
from icca.retrieval.hybrid_retriever import HybridRetriever, compute_jaccard_similarity
from icca.agents.retrieval_agent import RetrievalAgent


def test_stemmer_and_tokenizer():
    stemmed = simple_porter_stem("declarations")
    assert stemmed in ("declar", "declaration", "declarate")

    index = BM25Index(k1=1.5, b=0.75)
    tokens = index.tokenize("Customs declaration for HS Code 8471.30-00 imported!")
    assert "custom" in tokens
    assert "8471.30-00" in tokens
    assert "import" in tokens


def test_bm25_scoring_and_inverted_index():
    index = BM25Index(k1=1.5, b=0.75)
    chunks = [
        {"id": "doc1", "text": "China customs clearance guidelines for electronic computers"},
        {"id": "doc2", "text": "Agricultural tariffs and phytosanitary certificates for fruits"},
        {"id": "doc3", "text": "Electronic computer declaration process and customs duties"},
    ]
    index.index_chunks(chunks)

    assert index.total_docs == 3
    assert index.avgdl > 0

    # Search for electronics / computers
    results = index.search("electronic computers", top_k=2)
    assert len(results) == 2
    top_ids = [r["id"] for r in results]
    assert "doc1" in top_ids and "doc3" in top_ids

    # Search for agriculture
    agri_res = index.search("agricultural fruits", top_k=1)
    assert len(agri_res) == 1
    assert agri_res[0]["id"] == "doc2"

    # Test save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = Path(tmpdir) / "bm25.json"
        index.save(save_file)
        assert save_file.exists()

        new_index = BM25Index()
        new_index.load(save_file)
        assert new_index.total_docs == 3
        reloaded_res = new_index.search("electronic computers", top_k=2)
        assert [r["id"] for r in reloaded_res] == top_ids


def test_embeddings_manager_hash():
    em = EmbeddingsManager(provider="hash", dimension=384)
    assert em.dimension == 384
    assert em.backend_name == "hash"

    v1 = em.get_embedding("customs clearance procedures for machinery")
    v2 = em.get_embedding("machinery import customs clearance")
    v3 = em.get_embedding("chocolate chip cookie bakery recipe")

    assert len(v1) == 384
    # Unit norm check
    norm1 = math.sqrt(sum(x * x for x in v1))
    assert abs(norm1 - 1.0) < 1e-4

    # Semantic similarity check
    sim_related = em.cosine_similarity(v1, v2)
    sim_unrelated = em.cosine_similarity(v1, v3)
    assert sim_related > sim_unrelated
    assert sim_related > 0.5


def test_vector_index():
    vi = VectorIndex(dimension=4, metric="cosine")
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
    ]
    ids = ["v1", "v2", "v3"]
    metas = [{"title": "Doc 1"}, {"title": "Doc 2"}, {"title": "Doc 3"}]

    vi.add_vectors(vectors, ids, metas)
    assert len(vi) == 3

    # Query closest to v1
    res = vi.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(res) == 2
    assert res[0]["id"] == "v1"
    assert res[0]["score"] == 1.0
    assert res[1]["id"] == "v3"

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        v_file = Path(tmpdir) / "vec.json"
        vi.save(v_file)
        assert v_file.exists()

        loaded_vi = VectorIndex(dimension=4)
        loaded_vi.load(v_file)
        assert len(loaded_vi) == 3
        res_loaded = loaded_vi.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert res_loaded[0]["id"] == "v1"


def test_hybrid_retriever_rrf_and_weighted():
    hr = HybridRetriever()
    docs = [
        {
            "id": "c1",
            "text": "Certificate of Origin and commercial invoices are required for import clearance.",
            "metadata": {"doc_type": "guideline"},
        },
        {
            "id": "c2",
            "text": "Tariff classification under HS 8504 for electrical transformers is 5 percent.",
            "metadata": {"doc_type": "tariff"},
        },
        {
            "id": "c3",
            "text": "Sanitary inspections apply to imported poultry and dairy products.",
            "metadata": {"doc_type": "inspection"},
        },
    ]
    hr.index_documents(docs)

    # Test RRF
    rrf_results = hr.retrieve("commercial invoices certificate of origin", top_k=2, mode="rrf")
    assert len(rrf_results) > 0
    assert rrf_results[0]["id"] == "c1"
    assert "sparse_score" in rrf_results[0]
    assert "dense_score" in rrf_results[0]
    assert "rrf_score" in rrf_results[0]

    # Test Weighted
    weighted_results = hr.retrieve("electrical transformers tariff classification", top_k=2, mode="weighted")
    assert len(weighted_results) > 0
    assert weighted_results[0]["id"] == "c2"

    # Test Redundancy Removal
    duplicate_docs = [
        {"id": "dup1", "text": "Required documents: commercial invoices and certificate of origin"},
        {"id": "dup2", "text": "Required documents: commercial invoices and certificate of origin"},
    ]
    hr2 = HybridRetriever(redundancy_threshold=0.8)
    hr2.index_documents(duplicate_docs)
    dedup_results = hr2.retrieve("commercial invoices certificate of origin", top_k=2)
    # Should deduplicate down to 1 candidate
    assert len(dedup_results) == 1


def test_retrieval_agent():
    agent = RetrievalAgent()
    docs = [
        {"id": "doc_a", "text": "Customs duties on luxury automobiles."},
        {"id": "doc_b", "text": "Export licenses for dual-use high tech goods."},
    ]

    # Action: index
    res_index = agent.execute({"action": "index", "payload": {"chunks": docs}})
    assert res_index["status"] == "success"
    assert res_index["data"]["indexed_count"] == 2

    # Action: search_sparse
    res_sparse = agent.execute({"action": "search_sparse", "payload": {"query": "automobiles", "top_k": 1}})
    assert res_sparse["status"] == "success"
    assert len(res_sparse["data"]["results"]) == 1
    assert res_sparse["data"]["results"][0]["id"] == "doc_a"

    # Action: search_dense
    res_dense = agent.execute({"action": "search_dense", "payload": {"query": "export licenses", "top_k": 1}})
    assert res_dense["status"] == "success"
    assert len(res_dense["data"]["results"]) == 1
    assert res_dense["data"]["results"][0]["id"] == "doc_b"

    # Action: search_hybrid
    res_hybrid = agent.execute({"action": "search_hybrid", "payload": {"query": "customs luxury", "top_k": 1}})
    assert res_hybrid["status"] == "success"
    assert len(res_hybrid["data"]["results"]) == 1
    assert res_hybrid["data"]["results"][0]["id"] == "doc_a"

    # Action: save & load
    with tempfile.TemporaryDirectory() as tmpdir:
        res_save = agent.execute({"action": "save", "payload": {"path": tmpdir}})
        assert res_save["status"] == "success"

        new_agent = RetrievalAgent()
        res_load = new_agent.execute({"action": "load", "payload": {"path": tmpdir}})
        assert res_load["status"] == "success"
        assert res_load["data"]["bm25_count"] == 2

    # Action: invalid
    res_invalid = agent.execute({"action": "nonexistent_action"})
    assert res_invalid["status"] == "error"


if __name__ == "__main__":
    print("Running test_stemmer_and_tokenizer...")
    test_stemmer_and_tokenizer()
    print("Running test_bm25_scoring_and_inverted_index...")
    test_bm25_scoring_and_inverted_index()
    print("Running test_embeddings_manager_hash...")
    test_embeddings_manager_hash()
    print("Running test_vector_index...")
    test_vector_index()
    print("Running test_hybrid_retriever_rrf_and_weighted...")
    test_hybrid_retriever_rrf_and_weighted()
    print("Running test_retrieval_agent...")
    test_retrieval_agent()
    print("\nALL RETRIEVAL & INDEXING TESTS PASSED SUCCESSFULLY!")
