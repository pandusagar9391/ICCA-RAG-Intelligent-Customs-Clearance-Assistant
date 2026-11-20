"""
Generation Module for ICCA-RAG.

Provides grounded response generation with source citations,
confidence scoring, and multi-provider LLM support (Gemini, OpenAI, Offline).
"""

from icca.generation.llm_client import UnifiedLLMClient, OfflineCustomsSynthesizer
from icca.generation.generator import ResponseGenerator

__all__ = [
    "UnifiedLLMClient",
    "OfflineCustomsSynthesizer",
    "ResponseGenerator",
]
