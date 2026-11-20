"""
Document Parser and Semantic Chunker module for ICCA-RAG.
Implements Algorithm 1 (Unified Multimodal Parsing Workflow) from the IEEE Access paper.
"""

from icca.document_parser.parser import (
    DocumentParser,
    ParsedDocument,
    ParsedElement
)

from icca.document_parser.chunker import (
    DocumentChunker,
    TextChunk
)

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "ParsedElement",
    "DocumentChunker",
    "TextChunk"
]
