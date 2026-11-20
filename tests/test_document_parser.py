"""
Unit and integration tests for ICCA-RAG Document Parsing and Ingestion Agent.
Tests DocumentParser (Algorithm 1), DocumentChunker, and DataIngestionAgent.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from icca.config import default_config
from icca.document_parser.parser import DocumentParser, ParsedDocument, ParsedElement
from icca.document_parser.chunker import DocumentChunker, TextChunk
from icca.agents.ingestion_agent import DataIngestionAgent


def test_document_parser_txt():
    """Test text parsing with titles and paragraphs."""
    parser = DocumentParser()
    sample_text = (
        "# CHAPTER 1: GENERAL PROVISIONS\n\n"
        "This is paragraph one detailing customs procedures.\n\n"
        "SECTION 2: DECLARATIONS\n\n"
        "This is paragraph two covering Bill of Entry filing.\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(sample_text)
        temp_path = f.name

    try:
        doc = parser.parse(temp_path)
        assert doc.doc_type == "txt"
        assert len(doc.elements) >= 3
        types = [e.element_type for e in doc.elements]
        assert "title" in types
        assert "paragraph" in types
    finally:
        os.remove(temp_path)


def test_document_parser_csv():
    """Test CSV parsing into formatted Markdown table."""
    parser = DocumentParser()
    sample_csv = "HS_Code,Description,Duty\n8471.30,Laptops,0%\n8429.51,Loaders,7.5%\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(sample_csv)
        temp_path = f.name

    try:
        doc = parser.parse(temp_path)
        assert doc.doc_type == "csv"
        assert len(doc.elements) == 1
        assert doc.elements[0].element_type == "table"
        table_md = doc.elements[0].content
        assert "| HS_Code | Description | Duty |" in table_md
        assert "| 8471.30 | Laptops | 0% |" in table_md
    finally:
        os.remove(temp_path)


def test_document_parser_docx_fallback():
    """Test fallback DOCX XML extraction without python-docx."""
    parser = DocumentParser()
    # Create a minimal valid docx (zipped xml)
    doc_xml = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Customs Clearance Procedure Overview</w:t></w:r></w:p>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Col1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Col2</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Val1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Val2</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>"""

    with tempfile.NamedTemporaryFile("wb", suffix=".docx", delete=False) as f:
        temp_docx = f.name

    try:
        with zipfile.ZipFile(temp_docx, "w") as z:
            z.writestr("word/document.xml", doc_xml)

        doc = parser.parse(temp_docx)
        assert doc.doc_type == "docx"
        elem_types = [e.element_type for e in doc.elements]
        assert "paragraph" in elem_types
        assert "table" in elem_types
        full_text = doc.get_full_text()
        assert "Customs Clearance Procedure Overview" in full_text
        assert "| Col1 | Col2 |" in full_text
    finally:
        os.remove(temp_docx)


def test_document_parser_image_ocr_fallback():
    """Test image parser graceful fallback when tesseract binary is not installed."""
    parser = DocumentParser()
    with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as f:
        # 1x1 dummy png byte header
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        temp_img = f.name

    try:
        doc = parser.parse(temp_img)
        assert doc.doc_type == "img"
        assert len(doc.elements) == 1
        assert doc.elements[0].element_type == "image_ocr"
        assert "OCR" in doc.elements[0].content
    finally:
        os.remove(temp_img)


def test_document_chunker_short_document():
    """Test Algorithm 1 requirement: short documents are kept as single units."""
    chunker = DocumentChunker(chunk_size=500, length_threshold=200)
    parsed_doc = ParsedDocument(
        doc_id="short_doc",
        file_path="short.txt",
        doc_type="txt",
        elements=[ParsedElement(element_type="paragraph", content="Short notice on tariff duty for HS 8471.")],
        raw_text="Short notice on tariff duty for HS 8471."
    )
    chunks = chunker.chunk_document(parsed_doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "short_doc_c0"
    assert chunks[0].metadata["is_short_doc"] is True
    assert "HS_8471" in chunks[0].metadata["tags"]


def test_document_chunker_tag_extraction():
    """Test tag extraction for HS codes and customs compliance terms."""
    chunker = DocumentChunker()
    text = (
        "Under Section 46 of Customs Act, 1962, the importer filed Bill of Entry for HS 8429.51.00 "
        "and HS 8471.30.10. Basic Customs Duty BCD and IGST apply. Requires BIS registration and MSDS."
    )
    tags = chunker.extract_tags(text)
    assert "HS_8429.51.00" in tags
    assert "HS_8471.30.10" in tags
    assert "Bill of Entry" in tags
    assert "BCD" in tags
    assert "IGST" in tags
    assert "BIS" in tags
    assert "MSDS" in tags
    assert "HS_1962" not in tags  # Ensure year 1962 is not mistakenly tagged as HS code


def test_ingestion_agent_execute_ingest_file():
    """Test DataIngestionAgent action 'ingest_file' on a sample document."""
    agent = DataIngestionAgent()
    sample_file = default_config.sample_docs_dir / "customs_machinery_imports_guide.txt"
    assert sample_file.exists(), f"Sample doc not found: {sample_file}"

    task = {
        "action": "ingest_file",
        "payload": {"file_path": str(sample_file)}
    }
    result = agent.execute(task)
    assert result["status"] == "success"
    data = result["data"]
    assert data["doc_id"] == "customs_machinery_imports_guide"
    assert data["num_chunks"] > 0
    first_chunk = data["chunks"][0]
    assert "chunk_id" in first_chunk
    assert "text" in first_chunk
    assert "metadata" in first_chunk
    assert "tags" in first_chunk["metadata"]


def test_ingestion_agent_execute_ingest_directory():
    """Test DataIngestionAgent action 'ingest_directory' on sample_docs."""
    agent = DataIngestionAgent()
    task = {
        "action": "ingest_directory",
        "payload": {
            "directory_path": str(default_config.sample_docs_dir),
            "recursive": False
        }
    }
    result = agent.execute(task)
    assert result["status"] == "success"
    data = result["data"]
    assert data["total_files_processed"] == 13
    assert data["total_chunks"] >= 100
    doc_ids = [d["doc_id"] for d in data["documents"]]
    assert "customs_machinery_imports_guide" in doc_ids
    assert "customs_tariff_schedule_ch84_85" in doc_ids
    assert "customs_chemical_hazmat_regulations" in doc_ids
    assert "customs_dual_use_export_control" in doc_ids
    assert "customs_clearance_standard_sop" in doc_ids
    assert "customs_medical_devices_pharma" in doc_ids
    assert "customs_food_agriculture_imports" in doc_ids
    assert "customs_textiles_apparel" in doc_ids
    assert "customs_export_procedures" in doc_ids
    assert "customs_tariff_schedule_ch28_29_30_90" in doc_ids


def test_ingestion_agent_parse_only_and_chunk_text():
    """Test 'parse_only' and 'chunk_text' actions."""
    agent = DataIngestionAgent()
    sample_file = default_config.sample_docs_dir / "customs_tariff_schedule_ch84_85.csv"

    # Test parse_only
    p_task = {"action": "parse_only", "payload": {"file_path": str(sample_file)}}
    p_res = agent.execute(p_task)
    assert p_res["status"] == "success"
    assert p_res["data"]["doc_type"] == "csv"
    assert len(p_res["data"]["elements"]) >= 1

    # Test chunk_text
    raw_text = "This is a quick test of raw text chunking for customs clearance."
    c_task = {"action": "chunk_text", "payload": {"text": raw_text, "doc_id": "raw_test"}}
    c_res = agent.execute(c_task)
    assert c_res["status"] == "success"
    assert c_res["data"]["num_chunks"] == 1
