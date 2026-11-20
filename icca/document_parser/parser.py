"""
Unified Multimodal Document Parser for ICCA-RAG.
Implements Algorithm 1 (Unified Multimodal Parsing Workflow) from the IEEE Access paper.

Supports:
- DOCX (.docx) with python-docx and fallback XML parser
- PDF (.pdf) with PyMuPDF / PyPDF and stream fallback
- CSV (.csv) & Excel (.xlsx) with pandas / openpyxl / standard library csv & zipfile XML
- Plain Text (.txt, .md)
- Images (.png, .jpg, .jpeg, .tiff, .bmp) with Tesseract OCR & graceful fallback
"""

import os
import re
import csv
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

# Optional dependencies handled gracefully
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


@dataclass
class ParsedElement:
    """Represents a single structural element parsed from a document."""
    element_type: str  # 'title', 'paragraph', 'table', 'image_ocr', 'raw'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Structured representation of a parsed multimodal document."""
    doc_id: str
    file_path: str
    doc_type: str
    elements: List[ParsedElement] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Return the combined text of all elements."""
        if self.raw_text:
            return self.raw_text
        return "\n\n".join(elem.content for elem in self.elements if elem.content.strip())


class DocumentParser:
    """
    Implements Algorithm 1 (Unified Multimodal Parsing Workflow) from ICCA-RAG.
    
    Transforms diverse customs document formats (DOCX, PDF, CSV, Excel, TXT, Images)
    into structured text units with preserved table layouts and metadata.
    """

    SUPPORTED_EXTENSIONS = {
        "docx": [".docx"],
        "pdf": [".pdf"],
        "excel": [".xlsx", ".xls"],
        "csv": [".csv", ".tsv"],
        "txt": [".txt", ".text", ".md", ".log"],
        "img": [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]
    }

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the parser with optional OCR configuration.
        
        Args:
            tesseract_cmd: Optional path to tesseract binary if not in PATH.
        """
        self.tesseract_available = False
        if HAS_PYTESSERACT and HAS_PIL:
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            try:
                # Test if tesseract binary can actually be invoked
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
            except Exception:
                self.tesseract_available = False

    def get_document_type(self, file_path: Union[str, Path]) -> str:
        """
        Identify document type based on file extension.
        
        Algorithm 1: type <- GetDocumentType(D)
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        for doc_type, extensions in self.SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return doc_type
        return "txt"  # Default fallback

    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Execute unified multimodal parsing on the given document.
        
        Algorithm 1 Unified Multimodal Parsing Workflow:
        1: Initialize T <- empty set
        2: Identify document type: type <- GetDocumentType(D)
        3: if type = "DOCX" then ParseDocx(D)
        4: else if type = "PDF" then ParsePdf(D)
        5: else if type = "IMG" then OCR(D)
        6: else if type = "Excel" / "CSV" then ParseExcel/ParseCSV(D)
        7: else ParseTxt(D)
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        doc_type = self.get_document_type(path)
        doc_id = path.stem

        file_metadata = {
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "file_extension": path.suffix.lower(),
            "absolute_path": str(path),
        }

        if doc_type == "docx":
            parsed_doc = self._parse_docx(path, doc_id)
        elif doc_type == "pdf":
            parsed_doc = self._parse_pdf(path, doc_id)
        elif doc_type == "csv":
            parsed_doc = self._parse_csv(path, doc_id)
        elif doc_type == "excel":
            parsed_doc = self._parse_excel(path, doc_id)
        elif doc_type == "img":
            parsed_doc = self._parse_image(path, doc_id)
        else:
            parsed_doc = self._parse_txt(path, doc_id)

        # Merge base file metadata
        parsed_doc.metadata.update(file_metadata)
        if not parsed_doc.raw_text:
            parsed_doc.raw_text = parsed_doc.get_full_text()

        return parsed_doc

    # --------------------------------------------------------------------------
    # 1. DOCX Parser (Algorithm 1, Lines 3-17)
    # --------------------------------------------------------------------------
    def _parse_docx(self, path: Path, doc_id: str) -> ParsedDocument:
        """
        Parse DOCX: extract paragraphs, titles, tables formatted as markdown, and images.
        Uses python-docx if available, with native XML fallback.
        """
        elements: List[ParsedElement] = []

        if HAS_DOCX:
            try:
                doc = docx.Document(path)
                # Iterate through blocks (paragraphs and tables)
                for element in doc.element.body:
                    tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
                    if tag == "p":
                        # Check paragraph style
                        p = docx.text.paragraph.Paragraph(element, doc)
                        text = p.text.strip()
                        if text:
                            style_name = p.style.name.lower() if p.style else ""
                            is_title = "heading" in style_name or "title" in style_name
                            elem_type = "title" if is_title else "paragraph"
                            elements.append(ParsedElement(
                                element_type=elem_type,
                                content=text,
                                metadata={"style": p.style.name if p.style else "Normal"}
                            ))
                    elif tag == "tbl":
                        t = docx.table.Table(element, doc)
                        rows_data = []
                        for row in t.rows:
                            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                            rows_data.append(row_cells)
                        if rows_data:
                            headers = rows_data[0]
                            data_rows = rows_data[1:] if len(rows_data) > 1 else []
                            md_table = self.format_table_as_markdown(headers, data_rows)
                            elements.append(ParsedElement(
                                element_type="table",
                                content=md_table,
                                metadata={"rows": len(rows_data), "cols": len(headers)}
                            ))
                
                # Check for inline media images
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        img_bytes = rel.target_part.blob
                        img_text = self._ocr_bytes(img_bytes, rel.target_ref)
                        elements.append(ParsedElement(
                            element_type="image_ocr",
                            content=img_text,
                            metadata={"image_ref": rel.target_ref}
                        ))

                return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="docx", elements=elements)
            except Exception:
                # Fall back to zipfile XML parser
                pass

        # Fallback: Native python zipfile + XML parser for DOCX
        return self._parse_docx_fallback(path, doc_id)

    def _parse_docx_fallback(self, path: Path, doc_id: str) -> ParsedDocument:
        """Pure-Python fallback for DOCX extraction using zipfile and xml.etree."""
        elements: List[ParsedElement] = []
        try:
            with zipfile.ZipFile(path) as z:
                # Read document.xml
                xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

                for child in root.iter():
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "p":
                        texts = [node.text for node in child.iter() if node.tag.endswith("t") and node.text]
                        para_text = "".join(texts).strip()
                        if para_text:
                            # Check if pStyle is heading
                            pStyle = child.find(".//w:pStyle", ns)
                            style_val = pStyle.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "") if pStyle is not None else ""
                            elem_type = "title" if "heading" in style_val.lower() or "title" in style_val.lower() else "paragraph"
                            elements.append(ParsedElement(
                                element_type=elem_type,
                                content=para_text,
                                metadata={"parser": "fallback_xml", "style": style_val}
                            ))
                    elif tag == "tbl":
                        rows = []
                        for tr in child.findall("w:tr", ns):
                            cells = []
                            for tc in tr.findall("w:tc", ns):
                                cell_texts = [node.text for node in tc.iter() if node.tag.endswith("t") and node.text]
                                cells.append("".join(cell_texts).strip().replace("\n", " "))
                            if cells:
                                rows.append(cells)
                        if rows:
                            headers = rows[0]
                            data_rows = rows[1:] if len(rows) > 1 else []
                            elements.append(ParsedElement(
                                element_type="table",
                                content=self.format_table_as_markdown(headers, data_rows),
                                metadata={"parser": "fallback_xml"}
                            ))
        except Exception as e:
            elements.append(ParsedElement(
                element_type="raw",
                content=f"[Error reading DOCX file: {e}]",
                metadata={"error": str(e)}
            ))

        return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="docx", elements=elements)

    # --------------------------------------------------------------------------
    # 2. PDF Parser (Algorithm 1, Lines 18-27)
    # --------------------------------------------------------------------------
    def _parse_pdf(self, path: Path, doc_id: str) -> ParsedDocument:
        """
        Parse PDF: extract page text and structure, plus embedded image OCR.
        Tries PyMuPDF (fitz) -> PyPDF -> Pure Python stream fallback.
        """
        elements: List[ParsedElement] = []

        # Try PyMuPDF (fitz)
        if HAS_PYMUPDF:
            try:
                doc = fitz.open(path)
                for page_idx, page in enumerate(doc):
                    page_text = page.get_text().strip()
                    if page_text:
                        elements.append(ParsedElement(
                            element_type="paragraph",
                            content=page_text,
                            metadata={"page_num": page_idx + 1, "engine": "pymupdf"}
                        ))
                    # Check for page images
                    image_list = page.get_images(full=True)
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_text = self._ocr_bytes(img_bytes, f"page_{page_idx+1}_img_{img_idx+1}")
                        elements.append(ParsedElement(
                            element_type="image_ocr",
                            content=img_text,
                            metadata={"page_num": page_idx + 1, "image_index": img_idx + 1}
                        ))
                return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="pdf", elements=elements)
            except Exception:
                pass

        # Try PyPDF
        if HAS_PYPDF:
            try:
                reader = pypdf.PdfReader(str(path))
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        elements.append(ParsedElement(
                            element_type="paragraph",
                            content=text.strip(),
                            metadata={"page_num": page_idx + 1, "engine": "pypdf"}
                        ))
                if elements:
                    return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="pdf", elements=elements)
            except Exception:
                pass

        # Pure Python fallback for PDF (extract visible text streams)
        return self._parse_pdf_fallback(path, doc_id)

    def _parse_pdf_fallback(self, path: Path, doc_id: str) -> ParsedDocument:
        """Pure-Python basic text stream extractor for PDF files."""
        elements: List[ParsedElement] = []
        try:
            with open(path, "rb") as f:
                content = f.read()

            # Find uncompressed streams or text strings in parentheses
            bt_matches = re.findall(rb"BT\s*(.*?)\s*ET", content, re.DOTALL)
            extracted_lines = []
            for block in bt_matches:
                text_parts = re.findall(rb"\((.*?)\)\s*(?:Tj|'|\")", block)
                for part in text_parts:
                    try:
                        decoded = part.decode("latin1", errors="ignore")
                        if decoded.strip():
                            extracted_lines.append(decoded.strip())
                    except Exception:
                        continue

            if extracted_lines:
                full_text = " ".join(extracted_lines)
                elements.append(ParsedElement(
                    element_type="paragraph",
                    content=full_text,
                    metadata={"engine": "fallback_stream_extractor"}
                ))
            else:
                elements.append(ParsedElement(
                    element_type="raw",
                    content=f"[PDF content extracted for '{path.name}'. PyMuPDF or PyPDF recommended for full extraction.]",
                    metadata={"engine": "fallback_placeholder"}
                ))
        except Exception as e:
            elements.append(ParsedElement(
                element_type="raw",
                content=f"[Error reading PDF: {e}]",
                metadata={"error": str(e)}
            ))

        return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="pdf", elements=elements)

    # --------------------------------------------------------------------------
    # 3. Image Parser (Algorithm 1, Lines 28-30)
    # --------------------------------------------------------------------------
    def _parse_image(self, path: Path, doc_id: str) -> ParsedDocument:
        """
        Parse image: Apply Tesseract OCR if available, gracefully fallback to clean placeholder.
        """
        text = self._ocr_file(path)
        element = ParsedElement(
            element_type="image_ocr",
            content=text,
            metadata={"source": "image_ocr", "ocr_available": self.tesseract_available}
        )
        return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="img", elements=[element])

    def _ocr_file(self, path: Path) -> str:
        """OCR helper for image files."""
        if self.tesseract_available and HAS_PIL:
            try:
                img = Image.open(path)
                extracted = pytesseract.image_to_string(img).strip()
                if extracted:
                    return extracted
            except Exception:
                pass
        return f"[OCR Image Content: File '{path.name}' parsed. (Tesseract OCR binary not active)]"

    def _ocr_bytes(self, img_bytes: bytes, identifier: str) -> str:
        """OCR helper for image bytes embedded in documents."""
        if self.tesseract_available and HAS_PIL:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                extracted = pytesseract.image_to_string(img).strip()
                if extracted:
                    return extracted
            except Exception:
                pass
        return f"[OCR Embedded Image: '{identifier}' parsed. (Tesseract OCR binary not active)]"

    # --------------------------------------------------------------------------
    # 4. CSV Parser (Algorithm 1, Lines 31-35)
    # --------------------------------------------------------------------------
    def _parse_csv(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parse CSV or TSV: extract tabular data and format as GitHub-flavored Markdown."""
        elements: List[ParsedElement] = []
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

        if HAS_PANDAS:
            try:
                df = pd.read_csv(path, sep=delimiter, dtype=str).fillna("")
                headers = list(df.columns)
                rows = df.values.tolist()
                md_table = self.format_table_as_markdown(headers, rows)
                elements.append(ParsedElement(
                    element_type="table",
                    content=md_table,
                    metadata={"rows": len(rows), "cols": len(headers), "engine": "pandas"}
                ))
                return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="csv", elements=elements)
            except Exception:
                pass

        # Standard library CSV fallback
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = [row for row in reader if any(cell.strip() for cell in row)]
            if rows:
                headers = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []
                md_table = self.format_table_as_markdown(headers, data_rows)
                elements.append(ParsedElement(
                    element_type="table",
                    content=md_table,
                    metadata={"rows": len(data_rows), "cols": len(headers), "engine": "csv_builtin"}
                ))
        except Exception as e:
            elements.append(ParsedElement(
                element_type="raw",
                content=f"[Error parsing CSV: {e}]",
                metadata={"error": str(e)}
            ))

        return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="csv", elements=elements)

    # --------------------------------------------------------------------------
    # 5. Excel Parser (Algorithm 1, Lines 31-35)
    # --------------------------------------------------------------------------
    def _parse_excel(self, path: Path, doc_id: str) -> ParsedDocument:
        """
        Parse Excel (.xlsx, .xls): Extract all sheets and format as Markdown tables.
        Uses pandas / openpyxl if available, with native zipfile XML fallback.
        """
        elements: List[ParsedElement] = []

        if HAS_PANDAS:
            try:
                excel_file = pd.ExcelFile(path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, dtype=str).fillna("")
                    if not df.empty:
                        headers = list(df.columns)
                        rows = df.values.tolist()
                        md_table = self.format_table_as_markdown(headers, rows)
                        elements.append(ParsedElement(
                            element_type="table",
                            content=f"### Sheet: {sheet_name}\n\n{md_table}",
                            metadata={"sheet_name": sheet_name, "rows": len(rows), "cols": len(headers)}
                        ))
                if elements:
                    return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="excel", elements=elements)
            except Exception:
                pass

        # Fallback for .xlsx using zipfile + XML
        return self._parse_xlsx_fallback(path, doc_id)

    def _parse_xlsx_fallback(self, path: Path, doc_id: str) -> ParsedDocument:
        """Pure-Python fallback for XLSX extraction using zipfile and sharedStrings.xml."""
        elements: List[ParsedElement] = []
        try:
            with zipfile.ZipFile(path) as z:
                # 1. Read shared strings if available
                shared_strings = []
                if "xl/sharedStrings.xml" in z.namelist():
                    ss_xml = z.read("xl/sharedStrings.xml")
                    ss_root = ET.fromstring(ss_xml)
                    for si in ss_root.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                        text_nodes = [t.text for t in si.iter() if t.tag.endswith("t") and t.text]
                        shared_strings.append("".join(text_nodes))

                # 2. Read first worksheet
                sheet_files = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

                for sheet_file in sheet_files:
                    sheet_name = Path(sheet_file).stem
                    sheet_xml = z.read(sheet_file)
                    sheet_root = ET.fromstring(sheet_xml)

                    rows_data: List[List[str]] = []
                    sheet_data = sheet_root.find("s:sheetData", ns)
                    if sheet_data is not None:
                        for row_elem in sheet_data.findall("s:row", ns):
                            row_cells: List[str] = []
                            for c in row_elem.findall("s:c", ns):
                                cell_type = c.get("t")
                                val_elem = c.find("s:v", ns)
                                val = val_elem.text if val_elem is not None and val_elem.text is not None else ""
                                if cell_type == "s" and val.isdigit():
                                    idx = int(val)
                                    val = shared_strings[idx] if idx < len(shared_strings) else val
                                row_cells.append(val.strip().replace("\n", " "))
                            if any(row_cells):
                                rows_data.append(row_cells)

                    if rows_data:
                        headers = rows_data[0]
                        data_rows = rows_data[1:] if len(rows_data) > 1 else []
                        md_table = self.format_table_as_markdown(headers, data_rows)
                        elements.append(ParsedElement(
                            element_type="table",
                            content=f"### Sheet: {sheet_name}\n\n{md_table}",
                            metadata={"sheet_name": sheet_name, "parser": "fallback_xlsx_xml"}
                        ))
        except Exception as e:
            elements.append(ParsedElement(
                element_type="raw",
                content=f"[Error reading Excel file: {e}]",
                metadata={"error": str(e)}
            ))

        return ParsedDocument(doc_id=doc_id, file_path=str(path), doc_type="excel", elements=elements)

    # --------------------------------------------------------------------------
    # 6. Plain Text Parser
    # --------------------------------------------------------------------------
    def _parse_txt(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parse plain text documents (.txt, .md, .log), cleanly separating headings, lists, and paragraphs."""
        elements: List[ParsedElement] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            content = f"[Error reading file: {e}]"

        lines = content.splitlines()
        current_para: List[str] = []

        def flush_para():
            if current_para:
                para_text = "\n".join(current_para).strip()
                if para_text and para_text != "---":
                    elements.append(ParsedElement(
                        element_type="paragraph",
                        content=para_text,
                        metadata={"length": len(para_text)}
                    ))
                current_para.clear()

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str == "---":
                flush_para()
                continue

            # Check if line is a title or heading
            is_heading = (
                line_str.startswith("#")
                or (len(line_str) < 100 and line_str.isupper() and len(line_str.split()) < 10)
                or bool(re.match(r"^(SECTION|CHAPTER|PART|ARTICLE|STEP|STAGE)\s+[0-9IVXLCDM]+", line_str, re.IGNORECASE))
            )

            if is_heading:
                flush_para()
                elements.append(ParsedElement(
                    element_type="title",
                    content=line_str,
                    metadata={"heading_level": line_str.count("#") if line_str.startswith("#") else 1}
                ))
            else:
                current_para.append(line_str)

        flush_para()

        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(path),
            doc_type="txt",
            elements=elements,
            raw_text=content
        )

    # --------------------------------------------------------------------------
    # Utility: Markdown Table Formatter
    # --------------------------------------------------------------------------
    @staticmethod
    def format_table_as_markdown(headers: List[str], rows: List[List[str]]) -> str:
        """
        Format headers and tabular data into clean GitHub-flavored Markdown.
        Escapes pipes and normalizes column lengths.
        """
        if not headers and not rows:
            return ""

        max_cols = max(len(headers), max((len(r) for r in rows), default=0))
        if max_cols == 0:
            return ""

        # Normalize headers
        clean_headers = [str(h).replace("|", "\\|").strip() for h in headers]
        while len(clean_headers) < max_cols:
            clean_headers.append(f"Col_{len(clean_headers) + 1}")

        # Normalize rows
        clean_rows = []
        for row in rows:
            row_cells = [str(c).replace("|", "\\|").strip() for c in row]
            while len(row_cells) < max_cols:
                row_cells.append("")
            clean_rows.append(row_cells[:max_cols])

        # Build Markdown table
        header_line = "| " + " | ".join(clean_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * max_cols) + " |"
        row_lines = ["| " + " | ".join(r) + " |" for r in clean_rows]

        return "\n".join([header_line, sep_line] + row_lines)
