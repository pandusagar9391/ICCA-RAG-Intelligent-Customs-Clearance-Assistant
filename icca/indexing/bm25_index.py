"""
BM25 Inverted Index Implementation for ICCA-RAG.

Implements the sparse inverted index architecture and Okapi BM25 scoring
defined in Section IV-B and Equation (4) of the ICCA-RAG paper:
    Score(q, d) = sum_{t in q} IDF(t) * (TF(t, d) * (k1 + 1)) / (TF(t, d) + k1 * (1 - b + b * (|d| / avgdl)))

Features:
- Tokenization, lowercasing, punctuation stripping (preserving alphanumeric HS codes and identifiers).
- English stopword filtering.
- Lightweight pure-Python Porter-based suffix stemming for trade and legal terminology.
- Inverted index structure: term -> list of dicts {'id': doc_id, 'tf': count, 'length': length}.
- Robertson-Spärck Jones / Lucene IDF computation guaranteed non-negative.
- Persistent save and load via JSON.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


# Standard English stopwords (comprehensive set covering general and trade context)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't",
    "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def simple_porter_stem(word: str) -> str:
    """
    Lightweight rule-based suffix stemmer for English trade and general vocabulary.
    Reduces plurals and common inflections without external dependencies.
    """
    if len(word) <= 3:
        return word

    # Step 1: Plurals and participle suffixes
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]  # e.g., categories -> categor
    elif word.endswith("ss"):
        pass
    elif word.endswith("s") and not word.endswith("us") and not word.endswith("is"):
        word = word[:-1]

    # Step 2: Verb endings
    if word.endswith("eed") and len(word) > 4:
        word = word[:-1]
    elif word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if len(stem) >= 3:
            # Double consonant reduction: e.g., running -> run
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]
            word = stem
    elif word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if len(stem) >= 3:
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]
            word = stem

    # Step 3: Derivational suffixes
    suffixes = [
        ("ational", "ate"),
        ("ization", "ize"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("alism", "al"),
        ("ation", "ate"),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("ness", ""),
        ("ize", ""),
    ]
    for suf, rep in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[:-len(suf)] + rep
            break

    return word


class BM25Index:
    """
    Sparse Inverted Index with Okapi BM25 scoring for ICCA-RAG.
    
    Implements Section IV-B and Equation (4) of the ICCA-RAG paper:
        Score(q, d) = sum_{t in q} IDF(t) * (TF(t, d) * (k1 + 1)) / (TF(t, d) + k1 * (1 - b + b * (|d| / avgdl)))
    
    Attributes:
        k1 (float): Term frequency saturation parameter (default: 1.5).
        b (float): Document length normalization parameter (default: 0.75).
        stopwords (Set[str]): Set of stopwords to ignore.
        use_stemming (bool): Whether to apply stemming to terms.
        inverted_index (Dict[str, List[Dict[str, Any]]]):
            Mapping of term -> list of {'id': doc_id, 'tf': count, 'length': doc_len}.
        doc_metadata (Dict[str, Dict[str, Any]]): Mapping of doc_id -> metadata dict.
        doc_lengths (Dict[str, int]): Mapping of doc_id -> token length.
        total_docs (int): Total number of indexed documents.
        avgdl (float): Average document length across the collection.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Set[str]] = None,
        use_stemming: bool = True,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.use_stemming = use_stemming

        # Inverted index structure: term -> [{'id': doc_id, 'tf': int, 'length': int}]
        self.inverted_index: Dict[str, List[Dict[str, Any]]] = {}
        # Document store and metadata
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.total_docs: int = 0
        self.avgdl: float = 0.0

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize, normalize, and filter input text.
        
        Preserves numbers, HS codes (e.g., '8471.30', '8471'), and domain keywords.
        Applies lowercase conversion, stopword filtering, and optional stemming.
        
        Args:
            text: Raw input text.
            
        Returns:
            List of normalized string tokens.
        """
        if not text:
            return []

        # Convert to lower case
        text = text.lower()

        # Match words, alphanumeric strings, and HS codes with dots/hyphens (e.g., 8471.30, hs-code)
        raw_tokens = re.findall(r"[a-z0-9]+(?:[\.\-_][a-z0-9]+)*", text)

        tokens: List[str] = []
        for token in raw_tokens:
            # Strip outer punctuation
            token = token.strip(".-_")
            if not token:
                continue

            # Skip stopwords (do not skip purely numeric or code tokens like '01', '8471')
            if token in self.stopwords and not token.isdigit():
                continue

            # Apply stemming to non-numeric tokens
            if self.use_stemming and not token.isdigit():
                token = simple_porter_stem(token)

            tokens.append(token)

        return tokens

    def compute_idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency (IDF) for a given term.
        
        Uses Robertson-Spärck Jones IDF with non-negative guarantee (Lucene / standard BM25):
            IDF(t) = ln(1.0 + (N - DF(t) + 0.5) / (DF(t) + 0.5))
            
        Args:
            term: Normalized keyword term.
            
        Returns:
            IDF weight as float >= 0.
        """
        postings = self.inverted_index.get(term)
        if not postings:
            return 0.0

        df = len(postings)
        n = self.total_docs

        # Standard Okapi BM25 non-negative variant:
        # ln(1.0 + (N - DF + 0.5) / (DF + 0.5))
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return max(0.0, idf)

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single text chunk to the BM25 inverted index.
        
        Args:
            chunk_id: Unique identifier for the chunk.
            text: Raw text content of the chunk.
            metadata: Optional dictionary with chunk metadata (source, section, etc.).
        """
        # If document already exists, remove it first to avoid duplicate counting
        if chunk_id in self.doc_lengths:
            self._remove_chunk(chunk_id)

        tokens = self.tokenize(text)
        doc_len = len(tokens)

        # Count term frequencies in this document
        tf_counts: Dict[str, int] = {}
        for token in tokens:
            tf_counts[token] = tf_counts.get(token, 0) + 1

        # Update inverted index postings: term -> [{'id': doc_id, 'tf': count, 'length': doc_len}]
        for term, tf in tf_counts.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append({
                "id": chunk_id,
                "tf": tf,
                "length": doc_len,
            })

        # Record document metadata
        meta = dict(metadata or {})
        meta["id"] = chunk_id
        meta["text"] = text
        meta["length"] = doc_len
        self.doc_metadata[chunk_id] = meta
        self.doc_lengths[chunk_id] = doc_len

        # Update collection statistics
        self.total_docs = len(self.doc_lengths)
        if self.total_docs > 0:
            self.avgdl = sum(self.doc_lengths.values()) / float(self.total_docs)
        else:
            self.avgdl = 0.0

    def _remove_chunk(self, chunk_id: str) -> None:
        """Helper to remove a document and its postings from the index."""
        if chunk_id not in self.doc_lengths:
            return

        del self.doc_lengths[chunk_id]
        if chunk_id in self.doc_metadata:
            del self.doc_metadata[chunk_id]

        # Purge postings
        empty_terms = []
        for term, postings in self.inverted_index.items():
            self.inverted_index[term] = [p for p in postings if p["id"] != chunk_id]
            if not self.inverted_index[term]:
                empty_terms.append(term)

        for term in empty_terms:
            del self.inverted_index[term]

        self.total_docs = len(self.doc_lengths)
        self.avgdl = (sum(self.doc_lengths.values()) / float(self.total_docs)) if self.total_docs > 0 else 0.0

    def index_chunks(self, chunks: List[Any]) -> None:
        """
        Batch index a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries or strings.
                   Dict format: {'id': '...', 'text': '...', 'metadata': {...}}
                   String format: 'raw text' (auto-assigned chunk_0, chunk_1...)
        """
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or f"chunk_{i}")
                text = chunk.get("text") or chunk.get("content") or ""
                metadata = chunk.get("metadata", {})
                # Also capture any other top-level keys like source, section
                for k, v in chunk.items():
                    if k not in ("id", "chunk_id", "text", "content", "metadata"):
                        metadata[k] = v
            elif isinstance(chunk, str):
                chunk_id = f"chunk_{i}"
                text = chunk
                metadata = {}
            else:
                chunk_id = f"chunk_{i}"
                text = str(chunk)
                metadata = {}

            self.add_chunk(chunk_id, text, metadata)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the inverted index using Okapi BM25 scoring (Equation 4).
        
        Score(q, d) = sum_{t in q} IDF(t) * (TF(t, d) * (k1 + 1)) / (TF(t, d) + k1 * (1 - b + b * (|d| / avgdl)))
        
        Args:
            query: User search query string.
            top_k: Maximum number of candidate results to return.
            
        Returns:
            List of dicts: [{'id': ..., 'score': ..., 'text': ..., 'metadata': ...}]
            sorted in descending order of BM25 score.
        """
        if not query or self.total_docs == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Accumulate BM25 scores for matching documents
        doc_scores: Dict[str, float] = {}
        matched_terms_per_doc: Dict[str, List[str]] = {}

        for term in query_tokens:
            postings = self.inverted_index.get(term)
            if not postings:
                continue

            idf = self.compute_idf(term)
            if idf <= 0.0:
                continue

            for posting in postings:
                doc_id = posting["id"]
                tf = posting["tf"]
                doc_len = posting["length"]

                # BM25 Equation (4) denominator & numerator
                length_norm = 1.0 - self.b + self.b * (doc_len / (self.avgdl if self.avgdl > 0 else 1.0))
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * length_norm
                term_score = idf * (numerator / denominator)

                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score
                if doc_id not in matched_terms_per_doc:
                    matched_terms_per_doc[doc_id] = []
                matched_terms_per_doc[doc_id].append(term)

        if not doc_scores:
            return []

        # Sort documents by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            meta = self.doc_metadata.get(doc_id, {})
            results.append({
                "id": doc_id,
                "score": round(float(score), 4),
                "rank": rank,
                "text": meta.get("text", ""),
                "matched_terms": matched_terms_per_doc.get(doc_id, []),
                "metadata": {k: v for k, v in meta.items() if k not in ("text",)},
            })

        return results

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize the inverted index and document metadata to a JSON file.
        
        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "k1": self.k1,
            "b": self.b,
            "total_docs": self.total_docs,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "doc_metadata": self.doc_metadata,
            "inverted_index": self.inverted_index,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load the inverted index and document metadata from a JSON file.
        
        Args:
            path: Source file path.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.k1 = float(payload.get("k1", 1.5))
        self.b = float(payload.get("b", 0.75))
        self.total_docs = int(payload.get("total_docs", 0))
        self.avgdl = float(payload.get("avgdl", 0.0))
        self.doc_lengths = {str(k): int(v) for k, v in payload.get("doc_lengths", {}).items()}
        self.doc_metadata = payload.get("doc_metadata", {})
        self.inverted_index = payload.get("inverted_index", {})

    def clear(self) -> None:
        """Reset the index to an empty state."""
        self.inverted_index.clear()
        self.doc_metadata.clear()
        self.doc_lengths.clear()
        self.total_docs = 0
        self.avgdl = 0.0

    def __len__(self) -> int:
        return self.total_docs
