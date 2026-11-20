"""
Context Builder & Prompt Engineering Module for ICCA-RAG.

Implements Section IV-B.3 of the ICCA-RAG paper.
Merges Sparse Results (BM25) and Dense Results (Vector embeddings) with redundancy elimination,
injects multi-hop Knowledge Graph relationships/triples, and structures a grounded,
anti-hallucination prompt.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger("ContextBuilder")


def token_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate token-level Jaccard similarity between two chunk strings."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


class ContextBuilder:
    """
    Builds structured, grounded LLM prompts from hybrid retrieval and knowledge graph triples.
    Implements redundancy filtering and strict anti-hallucination prompt boundaries.
    """

    def __init__(self, redundancy_threshold: float = 0.80, max_chunks_per_modality: int = 5):
        self.redundancy_threshold = redundancy_threshold
        self.max_chunks_per_modality = max_chunks_per_modality

    def deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate chunks by chunk_id and textual Jaccard similarity.
        Retains higher-scoring chunk when near-duplicates are found.
        """
        thresh = threshold if threshold is not None else self.redundancy_threshold
        unique_chunks: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for chunk in chunks:
            cid = str(chunk.get("chunk_id", chunk.get("id", "")))
            text = chunk.get("text", chunk.get("content", "")).strip()
            if not text:
                continue

            if cid and cid in seen_ids:
                continue

            # Check similarity against already accepted chunks
            is_duplicate = False
            for accepted in unique_chunks:
                acc_text = accepted.get("text", accepted.get("content", ""))
                if token_jaccard_similarity(text, acc_text) >= thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                if cid:
                    seen_ids.add(cid)
                unique_chunks.append(chunk)

        return unique_chunks

    def merge_results(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        max_total: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merge sparse and dense retrieval results into a unified list,
        tracking modality tags ('Sparse', 'Dense', or 'Hybrid').
        """
        merged_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process sparse
        for s in sparse_results[:self.max_chunks_per_modality]:
            cid = str(s.get("chunk_id", s.get("id", len(merged_map))))
            merged_map[cid] = {
                **s,
                "modality": "Sparse (BM25)",
                "sparse_score": s.get("score", 0.0),
                "dense_score": None
            }

        # 2. Process dense
        for d in dense_results[:self.max_chunks_per_modality]:
            cid = str(d.get("chunk_id", d.get("id", f"dense_{len(merged_map)}")))
            if cid in merged_map:
                merged_map[cid]["modality"] = "Hybrid (Sparse + Dense)"
                merged_map[cid]["dense_score"] = d.get("score", 0.0)
            else:
                merged_map[cid] = {
                    **d,
                    "modality": "Dense (Semantic)",
                    "sparse_score": None,
                    "dense_score": d.get("score", 0.0)
                }

        # Deduplicate and sort
        merged_list = list(merged_map.values())
        return self.deduplicate_chunks(merged_list)[:max_total]

    def format_kg_triples(
        self,
        kg_triples: Optional[List[Any]] = None,
        kg_paths: Optional[List[str]] = None
    ) -> str:
        """Format Knowledge Graph triples and reasoning paths into logical statements."""
        lines: List[str] = []

        if kg_triples:
            for item in kg_triples:
                if isinstance(item, str):
                    lines.append(f"- {item}")
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    lines.append(f"- [{item[0]}] --{item[1]}--> [{item[2]}]")
                elif isinstance(item, dict):
                    subj = item.get("source", item.get("subject", item.get("entity1", "Entity")))
                    rel = item.get("relation", item.get("predicate", item.get("type", "RELATED_TO")))
                    obj = item.get("target", item.get("object", item.get("entity2", "Value")))
                    lines.append(f"- [{subj}] --{rel}--> [{obj}]")

        if kg_paths:
            lines.append("\nKnowledge Graph Traversal Paths:")
            for p in kg_paths:
                lines.append(f"  * {p}")

        if not lines:
            return "No matching Knowledge Graph relations found for this query."

        return "\n".join(lines)

    def format_chunk_section(self, chunks: List[Dict[str, Any]], title: str) -> str:
        """Format a list of document chunks with metadata and source citations."""
        if not chunks:
            return f"No {title.lower()} available."

        formatted_chunks: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("doc_id", chunk.get("source_doc", "Customs Regulation Doc"))
            chunk_id = chunk.get("chunk_id", chunk.get("id", f"C{i}"))
            score = chunk.get("score", chunk.get("sparse_score", chunk.get("dense_score", 0.0)))
            text = chunk.get("text", chunk.get("content", "")).strip()
            meta = chunk.get("metadata", {})
            page = meta.get("page_number", meta.get("page", ""))
            page_info = f", Page: {page}" if page else ""

            formatted_chunks.append(
                f"[{title} Document {i}] (Source: {doc_id}, Chunk ID: {chunk_id}{page_info}, Score: {score:.4f}):\n{text}"
            )

        return "\n\n".join(formatted_chunks)

    def build_prompt(
        self,
        query: str,
        sparse_results: Optional[List[Dict[str, Any]]] = None,
        dense_results: Optional[List[Dict[str, Any]]] = None,
        hybrid_results: Optional[List[Dict[str, Any]]] = None,
        kg_triples: Optional[List[Any]] = None,
        kg_paths: Optional[List[str]] = None
    ) -> str:
        """
        Construct the grounded, structured prompt following Section IV-B.3 of the ICCA-RAG paper.
        Enforces strict grounding, citing source documents, and zero hallucinations.
        """
        sparse = sparse_results or []
        dense = dense_results or []
        
        # If hybrid_results are supplied without explicit sparse/dense breakdown
        if not sparse and not dense and hybrid_results:
            sparse = [c for c in hybrid_results if "Sparse" in c.get("modality", "")]
            dense = [c for c in hybrid_results if "Dense" in c.get("modality", "")]
            if not sparse and not dense:
                dense = hybrid_results

        kg_text = self.format_kg_triples(kg_triples, kg_paths)
        sparse_text = self.format_chunk_section(sparse[:self.max_chunks_per_modality], "Sparse Retrieval (BM25)")
        dense_text = self.format_chunk_section(dense[:self.max_chunks_per_modality], "Dense Retrieval (Embeddings)")

        prompt = f"""### SYSTEM INSTRUCTION & GROUNDING MANDATE:
You are the Intelligent Customs Clearance Assistant (ICCA-RAG).
Your duty is to formulate an authoritative, legally accurate, and compliance-oriented response to the customs query.

CRITICAL ANTI-HALLUCINATION CONSTRAINTS:
1. Base your answer EXCLUSIVELY on the provided Knowledge Graph Relationships, Sparse Retrieval Results, and Dense Retrieval Results.
2. If the retrieved context does not contain sufficient details to verify an answer, explicitly output:
   "The provided customs documentation does not contain sufficient information to confirm this clearance requirement."
3. Do NOT invent, assume, or hallucinate HS codes, duty percentages, or clearance documents not supported by the context.
4. Always cite specific document names and chunk references for all requirements mentioned.
5. Organize your answer clearly with the following sections:
   - **Executive Clearance Summary**: Direct answer to the user's question.
   - **Applicable HS Codes & Tariff Rates**: Tariff line, chapter, Basic Customs Duty (BCD), and IGST where present.
   - **Mandatory Compliance & Document Checklist**: Checklist format (e.g. `[ ] Bill of Entry (Form I)`, `[ ] BIS Certificate`).
   - **Knowledge Graph Verification Trail**: Explanation of entity relationships connecting the goods to regulations.
   - **Source Document Citations**: Explicit references to the source files and chunk IDs used.

### USER QUERY:
{query}

### KNOWLEDGE GRAPH RELATIONSHIPS:
{kg_text}

### SPARSE RETRIEVAL RESULTS (Lexical / BM25):
{sparse_text}

### DENSE RETRIEVAL RESULTS (Semantic Embeddings):
{dense_text}

### GROUNDED RESPONSE:
"""
        return prompt

    def build_structured_context(
        self,
        query: str,
        sparse_results: Optional[List[Dict[str, Any]]] = None,
        dense_results: Optional[List[Dict[str, Any]]] = None,
        hybrid_results: Optional[List[Dict[str, Any]]] = None,
        kg_triples: Optional[List[Any]] = None,
        kg_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build both the prompt string and metadata dictionary for downstream consumption.
        """
        merged_chunks = self.merge_results(sparse_results or [], dense_results or [])
        prompt = self.build_prompt(
            query=query,
            sparse_results=sparse_results,
            dense_results=dense_results,
            hybrid_results=hybrid_results or merged_chunks,
            kg_triples=kg_triples,
            kg_paths=kg_paths
        )

        return {
            "query": query,
            "prompt": prompt,
            "merged_chunks": merged_chunks,
            "sparse_count": len(sparse_results or []),
            "dense_count": len(dense_results or []),
            "kg_triples_count": len(kg_triples or [])
        }
