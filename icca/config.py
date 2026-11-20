"""
Configuration module for ICCA-RAG.
Handles paths, model settings, retrieval hyperparameters, and system defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

@dataclass
class ICCAConfig:
    # Directories
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    sample_docs_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "sample_docs")
    storage_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "storage")
    kg_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "knowledge_graph")

    # Ingestion & Chunking (Algorithm 1)
    chunk_size: int = 500  # characters / words
    chunk_overlap: int = 50
    supported_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".pdf", ".docx", ".csv", ".xlsx", ".png", ".jpg"]
    )

    # Sparse Retrieval (BM25 parameters as in paper Section IV-B)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    sparse_top_k: int = 5

    # Dense Retrieval (Embeddings + HNSW / Vector Index)
    dense_top_k: int = 5
    embedding_dimension: int = 384
    dense_model_name: str = "BAAI/bge-m3"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")

    # Knowledge Graph
    kg_top_k_triples: int = 5
    kg_max_hops: int = 2

    # LLM Settings
    llM_provider: str = os.getenv("LLM_PROVIDER", "offline")  # "offline", "gemini", "openai"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    def ensure_directories(self):
        """Ensure all required runtime directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sample_docs_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.kg_dir.mkdir(parents=True, exist_ok=True)

# Default global instance
default_config = ICCAConfig()
default_config.ensure_directories()
