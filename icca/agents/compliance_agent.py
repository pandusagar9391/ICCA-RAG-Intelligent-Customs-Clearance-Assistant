"""
Compliance and Generation Agent for ICCA-RAG.

Specialized agent responsible for:
- Query reformulation and noise elimination (Section IV-B.1)
- Grounded prompt engineering and context building (Section IV-B.3)
- Multi-provider LLM response generation citing retrieved documents and KG paths
"""

import logging
from typing import Any, Dict, List, Optional, Union

from icca.agents.base_agent import BaseAgent
from icca.config import default_config, ICCAConfig
from icca.generation.generator import ResponseGenerator
from icca.generation.llm_client import UnifiedLLMClient
from icca.retrieval.context_builder import ContextBuilder
from icca.retrieval.query_reformulator import QueryReformulator

logger = logging.getLogger("ComplianceAgent")


class ComplianceAgent(BaseAgent):
    """
    Compliance and Response Generation Agent.
    Implements the generation, context engineering, and query reformulation stages of ICCA-RAG.
    """

    def __init__(
        self,
        name: str = "ComplianceAgent",
        role: str = "Customs Compliance & Generation Specialist",
        config: Optional[ICCAConfig] = None
    ):
        super().__init__(name=name, role=role)
        self.config = config or default_config

        # Submodules
        self.reformulator = QueryReformulator()
        self.context_builder = ContextBuilder(
            redundancy_threshold=0.80,
            max_chunks_per_modality=self.config.sparse_top_k
        )
        self.llm_client = UnifiedLLMClient(config=self.config)
        self.generator = ResponseGenerator(config=self.config, llm_client=self.llm_client)

        self.log(f"Initialized with active LLM provider '{self.llm_client.active_provider}'")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute delegated compliance, prompt-building, or generation task.

        Supported actions:
        - 'reformulate_query': cleans conversational noise and extracts trade entities.
        - 'build_context': formats sparse, dense, and KG context into a grounded prompt.
        - 'generate_answer': generates grounded response with citations and checklists.
        - 'set_provider': switches LLM provider ('offline', 'gemini', 'openai').
        - 'get_status': returns agent configuration and LLM readiness.

        Args:
            task: Task dictionary containing 'action' and parameters (either top-level or in 'payload').

        Returns:
            Dictionary with 'status' ('success' or 'error'), 'data', and 'message'.
        """
        action = task.get("action")
        payload = task.get("payload", {})

        if not action:
            return {
                "status": "error",
                "message": "Missing 'action' parameter in task.",
                "data": None
            }

        try:
            if action == "reformulate_query":
                return self._handle_reformulate_query(task, payload)
            elif action == "build_context":
                return self._handle_build_context(task, payload)
            elif action == "generate_answer":
                return self._handle_generate_answer(task, payload)
            elif action == "set_provider":
                return self._handle_set_provider(task, payload)
            elif action == "get_status":
                return self._handle_get_status()
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported action '{action}'. Valid actions: reformulate_query, build_context, generate_answer, set_provider, get_status.",
                    "data": None
                }
        except Exception as e:
            self.log(f"Error executing action '{action}': {str(e)}", level=logging.ERROR)
            return {
                "status": "error",
                "message": f"Failed to execute '{action}': {str(e)}",
                "data": None
            }

    def _get_field(self, task: Dict[str, Any], payload: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Helper to get a field from either the top-level task dict or the payload dict."""
        if key in task and task[key] is not None:
            return task[key]
        return payload.get(key, default)

    def _handle_reformulate_query(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Strip conversational fillers and synthesize optimized search query."""
        query = self._get_field(task, payload, "query", "")
        use_llm = self._get_field(task, payload, "use_llm", False)

        result = self.reformulator.reformulate(
            query=query,
            use_llm=use_llm,
            llm_client=self.llm_client if use_llm else None
        )

        return {
            "status": "success",
            "message": "Query reformulated successfully.",
            "data": result
        }

    def _handle_build_context(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Structure grounded anti-hallucination prompt from retrieved items."""
        query = self._get_field(task, payload, "query", "")
        sparse = self._get_field(task, payload, "sparse_results", []) or self._get_field(task, payload, "sparse", [])
        dense = self._get_field(task, payload, "dense_results", []) or self._get_field(task, payload, "dense", [])
        hybrid = (
            self._get_field(task, payload, "hybrid_results", [])
            or self._get_field(task, payload, "hybrid", [])
            or self._get_field(task, payload, "results", [])
        )
        kg_triples = self._get_field(task, payload, "kg_triples", [])
        kg_paths = self._get_field(task, payload, "kg_paths", [])

        context_data = self.context_builder.build_structured_context(
            query=query,
            sparse_results=sparse,
            dense_results=dense,
            hybrid_results=hybrid,
            kg_triples=kg_triples,
            kg_paths=kg_paths
        )

        return {
            "status": "success",
            "message": "Context built successfully.",
            "data": context_data
        }

    def _handle_generate_answer(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fact-grounded customs clearance answer with citations."""
        query = self._get_field(task, payload, "query", "")
        prompt = self._get_field(task, payload, "prompt", "")
        context_chunks = (
            self._get_field(task, payload, "context_chunks", [])
            or self._get_field(task, payload, "hybrid_results", [])
            or self._get_field(task, payload, "hybrid", [])
            or self._get_field(task, payload, "results", [])
        )
        kg_paths = self._get_field(task, payload, "kg_paths", [])

        # If prompt was not pre-built, build it on the fly
        if not prompt:
            built = self.context_builder.build_structured_context(
                query=query,
                hybrid_results=context_chunks,
                kg_paths=kg_paths
            )
            prompt = built["prompt"]

        gen_result = self.generator.generate_response(
            query=query,
            prompt=prompt,
            context_chunks=context_chunks,
            kg_paths=kg_paths
        )

        return {
            "status": "success",
            "message": "Answer generated successfully.",
            "data": gen_result
        }

    def _handle_set_provider(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Switch LLM provider (offline, gemini, openai)."""
        provider = self._get_field(task, payload, "provider", "offline")
        api_key = self._get_field(task, payload, "api_key", None)

        success = self.llm_client.set_provider(provider, api_key=api_key)
        if success:
            return {
                "status": "success",
                "message": f"LLM provider switched to '{self.llm_client.active_provider}'.",
                "data": {"provider": self.llm_client.active_provider}
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to switch LLM provider to '{provider}'. Check API key or configuration.",
                "data": {"provider": self.llm_client.active_provider}
            }

    def _handle_get_status(self) -> Dict[str, Any]:
        """Return readiness status."""
        llm_status = self.llm_client.get_status()
        return {
            "status": "success",
            "message": "ComplianceAgent is operational.",
            "data": {
                "agent": self.name,
                "role": self.role,
                "llm": llm_status,
                "config": {
                    "sparse_top_k": self.config.sparse_top_k,
                    "dense_top_k": self.config.dense_top_k,
                    "kg_max_hops": self.config.kg_max_hops
                }
            }
        }
