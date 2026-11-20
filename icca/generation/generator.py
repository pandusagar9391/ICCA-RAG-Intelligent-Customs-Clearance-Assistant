"""
Response Generator Module for ICCA-RAG.

Coordinates grounded LLM generation, extracts source citations with relevance scores,
identifies recommended HS codes, mandatory compliance documents, and calculates
an overall response confidence metric.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union

from icca.config import default_config, ICCAConfig
from icca.generation.llm_client import UnifiedLLMClient

logger = logging.getLogger("ResponseGenerator")


class ResponseGenerator:
    """
    Grounded Response Generator for ICCA-RAG.
    Produces structured answers citing retrieved documents and knowledge graph paths.
    """

    def __init__(
        self,
        config: Optional[ICCAConfig] = None,
        llm_client: Optional[UnifiedLLMClient] = None
    ):
        self.config = config or default_config
        self.llm_client = llm_client or UnifiedLLMClient(config=self.config)

    def extract_sources_from_chunks(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract structured source citations and metadata from context chunks."""
        sources: List[Dict[str, Any]] = []
        seen = set()

        for chunk in context_chunks:
            cid = str(chunk.get("chunk_id", chunk.get("id", "")))
            doc_id = chunk.get("doc_id", chunk.get("source_doc", "Customs Regulation"))
            key = f"{doc_id}::{cid}"
            if key in seen:
                continue
            seen.add(key)

            text = chunk.get("text", chunk.get("content", "")).strip()
            score = chunk.get("score", chunk.get("sparse_score", chunk.get("dense_score", 0.0))) or 0.0
            modality = chunk.get("modality", "Hybrid")
            meta = chunk.get("metadata", {})

            snippet = text[:200] + "..." if len(text) > 200 else text

            sources.append({
                "doc_id": doc_id,
                "chunk_id": cid,
                "relevance_score": round(float(score), 4),
                "modality": modality,
                "snippet": snippet,
                "page": meta.get("page_number", meta.get("page", None))
            })

        # Sort by relevance score descending
        sources.sort(key=lambda s: s["relevance_score"], reverse=True)
        return sources

    def extract_hs_codes(self, text: str) -> List[str]:
        """
        Extract genuine Harmonized System (HS) codes and chapter numbers from text.
        Strictly filters out scores, years, fee amounts, and random 4-digit numbers.
        """
        found: List[str] = []
        seen = set()

        # 1. HS code prefix: e.g. "HS 8429", "Heading 8429", "HS Code 8542.31"
        for m in re.findall(r"\b(?:HS(?:\s*Code)?|Heading|Tariff\s*Item|Subheading)\s*[:#-]?\s*([0-9]{4}(?:\.[0-9]{2}(?:\.[0-9]{2})?)?)\b", text, re.IGNORECASE):
            if m not in seen:
                seen.add(m)
                found.append(m)

        # 2. Dotted tariff items: e.g. 8429.51.00, 8542.31, 2902.11
        for m in re.findall(r"\b([0-9]{4}\.[0-9]{2}(?:\.[0-9]{2})?)\b", text):
            ch = int(m[:2])
            if 1 <= ch <= 98 and m not in seen:
                seen.add(m)
                found.append(m)

        # 3. Four-digit codes in valid chapter ranges if in tariff context
        for m in re.findall(r"\b([0-9]{4})\b", text):
            val = int(m)
            ch = int(m[:2])
            if 1900 <= val <= 2099 or val in (5000, 10000, 1500, 1000, 2000, 3000, 4000):
                continue
            if 1 <= ch <= 97:
                pattern = rf"(?:(?:HS|Heading|tariff|chapter|subheading|code|goods|classification|machinery|chemical|item)\s*(?:[^\w\n]{{1,15}}\s*)?{m}|{m}\s*(?:[^\w\n]{{1,15}}\s*)?(?:tariff|heading|subheading|code|excavator|machine|chemical|electronic))"
                if re.search(pattern, text, re.IGNORECASE):
                    if m not in seen:
                        seen.add(m)
                        found.append(m)

        # 4. Chapters: e.g. "Chapter 84", "Chapter 29"
        for ch in re.findall(r"\b(?:Chapter|Ch\.)\s*([0-9]{1,2})\b", text, re.IGNORECASE):
            ch_int = int(ch)
            if 1 <= ch_int <= 98:
                formatted = f"Chapter {ch.zfill(2)}"
                if formatted not in seen:
                    seen.add(formatted)
                    found.append(formatted)

        return found

    def extract_checklist(self, text: str) -> List[str]:
        """Extract checklist items from generated answer."""
        checklist: List[str] = []
        # Find lines starting with - [ ] or - [x] or * [ ]
        items = re.findall(r"[-*]\s*\[[ xX]?\]\s*(\*\*.*?\*\*|.*?)(?=\n|$)", text)
        for it in items:
            cleaned = it.replace("**", "").strip()
            if cleaned and cleaned not in checklist:
                checklist.append(cleaned)
        return checklist

    def compute_confidence(
        self,
        context_chunks: List[Dict[str, Any]],
        kg_paths: List[str],
        answer: str
    ) -> float:
        """
        Compute grounded confidence score (0.0 - 1.0) based on:
        - Quantity and top score of retrieved chunks
        - Knowledge Graph reasoning coverage
        - Absence of insufficient information disclaimers
        """
        if "does not contain sufficient information" in answer.lower():
            return 0.15

        base_score = 0.50

        # Retrieval score contribution
        if context_chunks:
            top_score = max([c.get("score", 0.0) or 0.0 for c in context_chunks], default=0.0)
            base_score += min(top_score * 0.25, 0.25)
            # Volume boost
            base_score += min(len(context_chunks) * 0.03, 0.12)

        # Knowledge Graph contribution
        if kg_paths:
            base_score += min(len(kg_paths) * 0.04, 0.10)

        # Structured content boost
        if "- [" in answer:
            base_score += 0.03

        return round(min(base_score, 0.98), 2)

    def generate_response(
        self,
        query: str,
        prompt: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        kg_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute grounded generation and return structured results.

        Args:
            query: User's query.
            prompt: Structured Section IV-B3 prompt.
            context_chunks: Retrieved document chunks.
            kg_paths: Knowledge Graph traversal paths.

        Returns:
            Dictionary with:
            - 'answer': formatted text
            - 'sources': list of source citations
            - 'hs_codes': detected HS codes
            - 'required_documents': extracted document checklist
            - 'confidence': confidence score (0.0 to 1.0)
            - 'provider': active LLM provider used
            - 'kg_paths': reasoning paths used
        """
        chunks = context_chunks or []
        paths = kg_paths or []

        # Generate answer through UnifiedLLMClient
        raw_answer = self.llm_client.generate(prompt)

        # Post-process and extract metadata
        sources = self.extract_sources_from_chunks(chunks)
        
        # Collect HS codes from query, prompt, and answer
        hs_codes = self.extract_hs_codes(f"{query}\n{raw_answer}")
        
        # Extract checklist
        checklist = self.extract_checklist(raw_answer)

        # Compute confidence
        confidence = self.compute_confidence(chunks, paths, raw_answer)

        return {
            "query": query,
            "answer": raw_answer,
            "sources": sources,
            "hs_codes": hs_codes,
            "required_documents": checklist,
            "confidence": confidence,
            "provider": self.llm_client.active_provider,
            "kg_paths": paths
        }
