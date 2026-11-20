"""
Central Coordinator for ICCA-RAG.
Orchestrates and delegates tasks across specialized sub-agents:
- IngestionAgent (Document parsing & chunking)
- RetrievalAgent (Dense & sparse hybrid search)
- KnowledgeGraphAgent (Graph store, entity extraction & multi-hop traversal)
- ComplianceAgent (Query reformulation, context building & LLM generation)
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from icca.config import default_config, ICCAConfig
from icca.agents.base_agent import BaseAgent

class CustomsClearanceCoordinator:
    """
    Central Coordinator orchestrating the entire ICCA-RAG + Knowledge Graph pipeline.
    Manages sub-agent lifecycles, data flow, and end-to-end question answering.
    """

    def __init__(self, config: Optional[ICCAConfig] = None):
        self.config = config or default_config
        self.config.ensure_directories()
        
        # Sub-agents (lazy-loaded or initialized)
        self.ingestion_agent: Optional[BaseAgent] = None
        self.retrieval_agent: Optional[BaseAgent] = None
        self.kg_agent: Optional[BaseAgent] = None
        self.compliance_agent: Optional[BaseAgent] = None
        
        self.is_initialized = False

    def initialize(self):
        """Initialize all sub-agents and load indexes/knowledge graph."""
        if self.is_initialized:
            return

        # 1. Ingestion Agent
        try:
            from icca.agents.ingestion_agent import IngestionAgent
            self.ingestion_agent = IngestionAgent(config=self.config)
        except Exception as e:
            print(f"[Coordinator] Warning initializing IngestionAgent: {e}")

        # 2. Retrieval Agent
        try:
            from icca.agents.retrieval_agent import RetrievalAgent
            self.retrieval_agent = RetrievalAgent(config=self.config)
        except Exception as e:
            print(f"[Coordinator] Warning initializing RetrievalAgent: {e}")

        # 3. Knowledge Graph Agent
        try:
            from icca.agents.kg_agent import KnowledgeGraphAgent
            self.kg_agent = KnowledgeGraphAgent(config=self.config)
        except Exception as e:
            print(f"[Coordinator] Warning initializing KnowledgeGraphAgent: {e}")

        # 4. Compliance & Generation Agent
        try:
            from icca.agents.compliance_agent import ComplianceAgent
            self.compliance_agent = ComplianceAgent(config=self.config)
        except Exception as e:
            print(f"[Coordinator] Warning initializing ComplianceAgent: {e}")

        self.is_initialized = True
        print("[Coordinator] All ICCA-RAG sub-agents registered and coordinator initialized.")

    def ingest_documents(self, source_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrate document ingestion:
        1. Delegates parsing and chunking to IngestionAgent.
        2. Delegates dual-indexing (BM25 sparse + dense vector) to RetrievalAgent.
        3. Optionally extracts entities to KnowledgeGraphAgent.
        """
        self.initialize()
        target = Path(source_path) if source_path else self.config.sample_docs_dir

        # Step 1: Parse and chunk
        if not self.ingestion_agent:
            return {"status": "error", "message": "IngestionAgent not available."}

        action = "ingest_directory" if target.is_dir() else "ingest_file"
        parse_res = self.ingestion_agent.execute({
            "action": action,
            "path": str(target),
            "file_path": str(target),
            "directory_path": str(target),
            "payload": {
                "path": str(target),
                "file_path": str(target),
                "directory_path": str(target)
            }
        })

        if parse_res.get("status") != "success":
            return parse_res

        chunks = parse_res.get("data", {}).get("chunks", [])
        if not chunks:
            return {"status": "warning", "message": "No chunks extracted.", "chunks_count": 0}

        # Step 2: Index in Sparse & Dense stores via RetrievalAgent
        index_res = {}
        if self.retrieval_agent:
            index_res = self.retrieval_agent.execute({
                "action": "index",
                "chunks": chunks
            })

        # Step 3: Populate KG entities if KnowledgeGraphAgent is available
        kg_res = {}
        if self.kg_agent:
            kg_res = self.kg_agent.execute({
                "action": "extract_and_add",
                "chunks": chunks
            })

        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {len(chunks)} text chunks.",
            "chunks_count": len(chunks),
            "indexing": index_res.get("data", {}),
            "kg": kg_res.get("data", {})
        }

    def process_query(self, user_query: str, top_k: int = 5, use_kg: bool = True) -> Dict[str, Any]:
        """
        End-to-end query answering workflow:
        1. Reformulate query (ComplianceAgent)
        2. Hybrid retrieval: Dense + Sparse (RetrievalAgent)
        3. Multi-hop knowledge graph traversal (KnowledgeGraphAgent)
        4. Context construction with anti-hallucination constraints (ComplianceAgent)
        5. Answer generation with citations (ComplianceAgent)
        """
        self.initialize()

        # Ensure documents are indexed in retrieval stores before query execution
        if (
            self.retrieval_agent
            and hasattr(self.retrieval_agent, "bm25_index")
            and len(self.retrieval_agent.bm25_index) == 0
        ):
            self.ingest_documents()

        # Step 1: Reformulate query
        reformulated_query = user_query
        keywords = []
        if self.compliance_agent:
            reform_res = self.compliance_agent.execute({
                "action": "reformulate_query",
                "query": user_query
            })
            if reform_res.get("status") == "success":
                data = reform_res.get("data", {})
                reformulated_query = data.get("reformulated_query", user_query)
                keywords = data.get("keywords", [])

        # Step 2: Hybrid Retrieval
        retrieval_results = {"sparse": [], "dense": [], "hybrid": []}
        if self.retrieval_agent:
            ret_res = self.retrieval_agent.execute({
                "action": "search_hybrid",
                "query": reformulated_query,
                "raw_query": user_query,
                "top_k": top_k
            })
            if ret_res.get("status") == "success":
                retrieval_results = ret_res.get("data", {})

        # Step 3: Knowledge Graph Traversal & Reasoning
        kg_context = []
        kg_paths = []
        if use_kg and self.kg_agent:
            kg_res = self.kg_agent.execute({
                "action": "get_context_triples",
                "query": user_query,
                "keywords": keywords,
                "max_hops": self.config.kg_max_hops
            })
            if kg_res.get("status") == "success":
                kg_data = kg_res.get("data", {})
                kg_context = kg_data.get("triples_text", [])
                kg_paths = kg_data.get("paths", [])

        # Step 4: Build Structured Context & Generate Answer
        if self.compliance_agent:
            # Build context prompt
            ctx_res = self.compliance_agent.execute({
                "action": "build_context",
                "query": user_query,
                "sparse_results": retrieval_results.get("sparse", []),
                "dense_results": retrieval_results.get("dense", []),
                "hybrid_results": retrieval_results.get("hybrid", []),
                "kg_triples": kg_context
            })
            structured_prompt = ctx_res.get("data", {}).get("prompt", "")

            # Generate Answer
            gen_res = self.compliance_agent.execute({
                "action": "generate_answer",
                "query": user_query,
                "prompt": structured_prompt,
                "context_chunks": retrieval_results.get("hybrid", []),
                "kg_paths": kg_paths
            })

            gen_data = gen_res.get("data", {})
            return {
                "status": "success",
                "query": user_query,
                "reformulated_query": reformulated_query,
                "keywords": keywords,
                "answer": gen_data.get("answer", "No answer generated."),
                "sources": gen_data.get("sources", []),
                "hs_codes": gen_data.get("hs_codes", []),
                "required_documents": gen_data.get("required_documents", []),
                "confidence": gen_data.get("confidence", 0.85),
                "provider": gen_data.get("provider", "offline"),
                "sparse_results": retrieval_results.get("sparse", []),
                "dense_results": retrieval_results.get("dense", []),
                "kg_triples": kg_context,
                "kg_paths": kg_paths,
                "structured_prompt": structured_prompt
            }

        return {
            "status": "error",
            "message": "ComplianceAgent is unavailable for generation.",
            "query": user_query
        }
