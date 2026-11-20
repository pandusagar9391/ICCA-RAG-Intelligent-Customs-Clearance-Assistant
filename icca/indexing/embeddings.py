"""
Flexible Embeddings Manager for ICCA-RAG.

Supports:
1. Sentence-Transformers (e.g., BAAI/bge-m3 or all-MiniLM-L6-v2).
2. Google Gemini embeddings API (models/text-embedding-004 or models/embedding-001).
3. OpenAI embeddings API (text-embedding-3-small or text-embedding-ada-002).
4. Deterministic Offline Subword Hash Dense Vectorizer (384 dimensions) as a fail-safe fallback
   that runs immediately on any OS/Python environment with ZERO API keys or heavy packages.
"""

import hashlib
import logging
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("EmbeddingsManager")


class DeterministicHashEmbedder:
    """
    Offline deterministic dense vectorizer using subword feature hashing (the hashing trick).
    
    Generates unit-normalized 384-dimensional dense vectors with:
    - Character n-grams (3-gram, 4-gram, 5-gram) for morphological and stem similarity.
    - Word-level token hashing with log-frequency scaling.
    - Pseudo-random projection spreading to prevent sparsity.
    - Unit L2-normalization so that dot-product equals cosine similarity.
    - 100% deterministic, offline, and zero-dependency.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _extract_features(self, text: str) -> Dict[str, float]:
        """Extract word tokens and character n-grams with frequency weights."""
        if not text:
            return {}

        text_lower = text.lower()
        # Find all alphanumeric tokens and identifiers
        words = re.findall(r"[a-z0-9]+(?:[\.\-_][a-z0-9]+)*", text_lower)
        if not words:
            return {}

        feature_counts: Dict[str, float] = {}

        for w in words:
            clean_w = w.strip(".-_")
            if not clean_w:
                continue

            # Word feature with weight 2.0
            word_key = f"w:{clean_w}"
            feature_counts[word_key] = feature_counts.get(word_key, 0.0) + 2.0

            # Subword character n-grams with boundary markers
            marked = f"<{clean_w}>"
            n = len(marked)
            for ngram_len in (3, 4, 5):
                if n >= ngram_len:
                    for i in range(n - ngram_len + 1):
                        ngram = marked[i : i + ngram_len]
                        ngram_key = f"ng:{ngram}"
                        feature_counts[ngram_key] = feature_counts.get(ngram_key, 0.0) + 1.0

        return feature_counts

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic 384-dimensional unit vector for text."""
        vec = [0.0] * self.dimension
        features = self._extract_features(text)
        if not features:
            return vec

        for feat, count in features.items():
            # Compute SHA-256 digest
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            # Primary index and sign
            idx1 = int.from_bytes(digest[:4], "big") % self.dimension
            sign1 = 1.0 if (digest[4] % 2 == 0) else -1.0
            weight = math.log1p(count)

            vec[idx1] += sign1 * weight

            # Secondary pseudo-random projection to distribute semantic energy across dense dims
            seed = int.from_bytes(digest[5:9], "big")
            rng = random.Random(seed)
            for _ in range(3):
                idx2 = rng.randint(0, self.dimension - 1)
                sign2 = 1.0 if rng.random() > 0.5 else -1.0
                vec[idx2] += sign2 * (weight * 0.35)

        # Apply L2 normalization
        norm_sq = sum(x * x for x in vec)
        if norm_sq > 0.0:
            inv_norm = 1.0 / math.sqrt(norm_sq)
            vec = [x * inv_norm for x in vec]

        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding of multiple texts."""
        return [self.embed_text(t) for t in texts]


class EmbeddingsManager:
    """
    Unified Embeddings Manager for ICCA-RAG.
    
    Dynamically loads and selects the best available embedding backend:
    - 'sentence_transformers': Local neural transformer models (e.g. BGE-M3, MiniLM).
    - 'gemini': Google Gemini embedding models (requires GEMINI_API_KEY).
    - 'openai': OpenAI embedding models (requires OPENAI_API_KEY).
    - 'hash': Built-in Deterministic Subword Hash Vectorizer (offline fallback).
    """

    def __init__(
        self,
        provider: str = "auto",
        model_name: Optional[str] = None,
        dimension: int = 384,
    ):
        self.dimension = dimension
        self.model_name = model_name
        self.provider = provider.lower() if provider else "auto"
        self._hash_embedder = DeterministicHashEmbedder(dimension=self.dimension)
        self._active_backend: Optional[str] = None
        self._model_instance: Any = None

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Probe and initialize the requested or optimal available backend."""
        # 1. Explicit or Auto Sentence-Transformers
        if self.provider in ("auto", "sentence_transformers", "local"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                chosen_model = self.model_name or "BAAI/bge-m3"
                logger.info(f"Loading local SentenceTransformer model: {chosen_model}")
                self._model_instance = SentenceTransformer(chosen_model)
                self._active_backend = "sentence_transformers"
                # Determine actual dimension
                try:
                    self.dimension = self._model_instance.get_sentence_embedding_dimension()
                except Exception:
                    pass
                logger.info(f"SentenceTransformer initialized successfully (dim={self.dimension}).")
                return
            except Exception as e:
                if self.provider == "sentence_transformers":
                    logger.warning(f"Failed to load sentence_transformers: {e}. Falling back to hash.")

        # 2. Gemini Embeddings
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if (self.provider in ("auto", "gemini")) and gemini_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=gemini_key)
                self._model_instance = genai
                self._active_backend = "gemini"
                self.model_name = self.model_name or "models/text-embedding-004"
                logger.info("Google Gemini embeddings backend initialized.")
                return
            except Exception as e:
                if self.provider == "gemini":
                    logger.warning(f"Failed to initialize Gemini embeddings: {e}. Falling back.")

        # 3. OpenAI Embeddings
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if (self.provider in ("auto", "openai")) and openai_key:
            try:
                from openai import OpenAI  # type: ignore

                self._model_instance = OpenAI(api_key=openai_key)
                self._active_backend = "openai"
                self.model_name = self.model_name or "text-embedding-3-small"
                logger.info("OpenAI embeddings backend initialized.")
                return
            except Exception as e:
                if self.provider == "openai":
                    logger.warning(f"Failed to initialize OpenAI embeddings: {e}. Falling back.")

        # 4. Fallback: Deterministic Offline Subword Hash
        self._active_backend = "hash"
        logger.info(
            f"Using Deterministic Subword Hash Embedder (offline, dim={self.dimension}). "
            "No API keys or external neural weights required."
        )

    @property
    def backend_name(self) -> str:
        """Returns the active backend name."""
        return self._active_backend or "hash"

    def get_embedding(self, text: str) -> List[float]:
        """
        Compute embedding vector for a single string.
        
        Args:
            text: Input text string.
            
        Returns:
            List of floats representing the dense embedding vector.
        """
        if not text:
            return [0.0] * self.dimension

        if self._active_backend == "sentence_transformers" and self._model_instance:
            try:
                vec = self._model_instance.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return [float(x) for x in vec.tolist()]
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding error: {e}. Using hash fallback.")
                return self._hash_embedder.embed_text(text)

        elif self._active_backend == "gemini" and self._model_instance:
            try:
                res = self._model_instance.embed_content(
                    model=self.model_name or "models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                embedding = res.get("embedding", {})
                values = embedding.get("values", [])
                if values:
                    # Normalize vector
                    norm = math.sqrt(sum(x * x for x in values))
                    if norm > 0:
                        return [float(x / norm) for x in values]
                    return [float(x) for x in values]
            except Exception as e:
                logger.warning(f"Gemini API error: {e}. Using hash fallback.")
                return self._hash_embedder.embed_text(text)

        elif self._active_backend == "openai" and self._model_instance:
            try:
                res = self._model_instance.embeddings.create(
                    input=text,
                    model=self.model_name or "text-embedding-3-small",
                )
                values = res.data[0].embedding
                norm = math.sqrt(sum(x * x for x in values))
                if norm > 0:
                    return [float(x / norm) for x in values]
                return [float(x) for x in values]
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}. Using hash fallback.")
                return self._hash_embedder.embed_text(text)

        # Default fallback
        return self._hash_embedder.embed_text(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a batch of strings.
        
        Args:
            texts: List of text strings.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if self._active_backend == "sentence_transformers" and self._model_instance:
            try:
                vecs = self._model_instance.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return [[float(x) for x in v] for v in vecs.tolist()]
            except Exception as e:
                logger.warning(f"Batch SentenceTransformer error: {e}. Using hash fallback.")
                return self._hash_embedder.embed_batch(texts)

        # For API models and fallback, map individual calls or use batch
        return [self.get_embedding(t) for t in texts]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two float vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            
        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        if len(vec1) != len(vec2) or not vec1:
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0

        return dot / (norm1 * norm2)
