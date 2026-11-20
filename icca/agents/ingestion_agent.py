"""
Data Ingestion Agent for ICCA-RAG.
Delegates document parsing and semantic chunking following Algorithm 1.
Inherits from BaseAgent.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from icca.agents.base_agent import BaseAgent
from icca.config import default_config, ICCAConfig
from icca.document_parser.parser import DocumentParser, ParsedDocument
from icca.document_parser.chunker import DocumentChunker, TextChunk


class DataIngestionAgent(BaseAgent):
    """
    Specialized agent responsible for multimodal document parsing and semantic chunking.
    Implements Algorithm 1 (Unified Multimodal Parsing Workflow) from the IEEE Access paper.
    """

    def __init__(
        self,
        name: str = "DataIngestionAgent",
        role: str = "Data Ingestion Specialist",
        config: Optional[ICCAConfig] = None
    ):
        super().__init__(name=name, role=role)
        self.config = config or default_config
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.log(f"Initialized with chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an ingestion task.
        
        Supported actions:
        - 'ingest_file': parses and chunks a single file.
        - 'ingest_directory': parses and chunks all supported files in a folder.
        - 'parse_only': parses a single file into ParsedDocument representation without chunking.
        - 'chunk_text': segments raw text string into TextChunks.
        
        Args:
            task: Dictionary with 'action', 'payload', and optional metadata.
            
        Returns:
            Dictionary with 'status' ('success' or 'error'), 'data', and 'message'.
        """
        action = task.get("action")
        raw_payload = task.get("payload", {})
        if isinstance(raw_payload, dict):
            payload = {**task, **raw_payload}
        else:
            payload = dict(task)

        if not action:
            return {
                "status": "error",
                "message": "Missing 'action' key in task dictionary.",
                "data": None
            }

        try:
            if action == "ingest_file":
                return self._handle_ingest_file(payload)
            elif action == "ingest_directory":
                return self._handle_ingest_directory(payload)
            elif action == "parse_only":
                return self._handle_parse_only(payload)
            elif action == "chunk_text":
                return self._handle_chunk_text(payload)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported action '{action}'. Supported actions: ingest_file, ingest_directory, parse_only, chunk_text.",
                    "data": None
                }
        except Exception as e:
            self.log(f"Error executing action '{action}': {str(e)}", level=40)  # logging.ERROR
            return {
                "status": "error",
                "message": f"Failed to execute '{action}': {str(e)}",
                "data": None
            }

    def _handle_ingest_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and chunk a single file."""
        file_path = payload.get("file_path") or payload.get("path") or payload.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing 'file_path' in payload.", "data": None}

        path = Path(file_path).resolve()
        self.log(f"Ingesting file: {path.name}")

        parsed_doc = self.parser.parse(path)
        chunks = self.chunker.chunk_document(parsed_doc)

        result_data = {
            "doc_id": parsed_doc.doc_id,
            "file_path": str(path),
            "doc_type": parsed_doc.doc_type,
            "num_elements": len(parsed_doc.elements),
            "num_chunks": len(chunks),
            "chunks": [chunk.to_dict() for chunk in chunks]
        }

        self.log(f"Successfully ingested '{path.name}': {len(chunks)} chunks generated.")
        return {
            "status": "success",
            "message": f"Parsed and chunked '{path.name}' into {len(chunks)} chunks.",
            "data": result_data
        }

    def _handle_ingest_directory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and chunk all supported documents in a directory."""
        dir_path_str = (
            payload.get("directory_path")
            or payload.get("path")
            or payload.get("file_path")
            or str(self.config.sample_docs_dir)
        )
        recursive = payload.get("recursive", False)
        dir_path = Path(dir_path_str).resolve()

        if not dir_path.exists() or not dir_path.is_dir():
            return {
                "status": "error",
                "message": f"Directory not found or is not a directory: {dir_path}",
                "data": None
            }

        self.log(f"Scanning directory: {dir_path} (recursive={recursive})")

        # Discover supported files
        supported_exts = set(ext.lower() for ext in self.config.supported_extensions)
        # Add all parser supported extensions
        for ext_list in DocumentParser.SUPPORTED_EXTENSIONS.values():
            supported_exts.update(ext_list)

        pattern = "**/*" if recursive else "*"
        matched_files = [
            f for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_exts
        ]

        # Sort for deterministic processing order
        matched_files.sort(key=lambda p: p.name)
        self.log(f"Found {len(matched_files)} supported files in {dir_path.name}")

        ingested_docs = []
        all_chunks = []
        failed_files = []

        for file_path in matched_files:
            try:
                parsed_doc = self.parser.parse(file_path)
                chunks = self.chunker.chunk_document(parsed_doc)
                chunk_dicts = [c.to_dict() for c in chunks]
                all_chunks.extend(chunk_dicts)

                ingested_docs.append({
                    "doc_id": parsed_doc.doc_id,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": parsed_doc.doc_type,
                    "num_elements": len(parsed_doc.elements),
                    "num_chunks": len(chunks)
                })
            except Exception as e:
                self.log(f"Warning: Failed to ingest file '{file_path.name}': {e}", level=30)
                failed_files.append({"file_name": file_path.name, "error": str(e)})

        summary_data = {
            "directory": str(dir_path),
            "total_files_processed": len(ingested_docs),
            "total_files_failed": len(failed_files),
            "total_chunks": len(all_chunks),
            "documents": ingested_docs,
            "failed_files": failed_files,
            "chunks": all_chunks
        }

        self.log(f"Directory ingestion completed: {len(ingested_docs)} files processed, {len(all_chunks)} total chunks.")
        return {
            "status": "success",
            "message": f"Ingested {len(ingested_docs)} documents yielding {len(all_chunks)} chunks.",
            "data": summary_data
        }

    def _handle_parse_only(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse document into ParsedDocument without chunking."""
        file_path = payload.get("file_path") or payload.get("path") or payload.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing 'file_path' in payload.", "data": None}

        path = Path(file_path).resolve()
        parsed_doc = self.parser.parse(path)

        elements_data = [
            {"type": elem.element_type, "content": elem.content, "metadata": elem.metadata}
            for elem in parsed_doc.elements
        ]

        return {
            "status": "success",
            "message": f"Parsed '{path.name}' successfully.",
            "data": {
                "doc_id": parsed_doc.doc_id,
                "file_path": str(path),
                "doc_type": parsed_doc.doc_type,
                "elements": elements_data,
                "raw_text": parsed_doc.raw_text,
                "metadata": parsed_doc.metadata
            }
        }

    def _handle_chunk_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk raw text string."""
        text = payload.get("text", "")
        doc_id = payload.get("doc_id", "raw_text_doc")
        doc_type = payload.get("doc_type", "txt")
        metadata = payload.get("metadata", {})

        chunks = self.chunker.chunk_text(text=text, doc_id=doc_id, doc_type=doc_type, metadata=metadata)
        return {
            "status": "success",
            "message": f"Created {len(chunks)} chunks from text input.",
            "data": {
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "chunks": [c.to_dict() for c in chunks]
            }
        }

# Alias for coordinator compatibility
IngestionAgent = DataIngestionAgent


# Alias for compatibility with coordinator imports
IngestionAgent = DataIngestionAgent
