"""
Semantic Section & Paragraph Chunker for ICCA-RAG.
Implements the document segmentation phase of Algorithm 1 from the IEEE Access paper.

Segments parsed documents into structured text units $t_{i,j}$ with metadata:
- chunk_id
- doc_id
- text
- section_num
- paragraph_num
- token_count
- metadata (doc_type, tags, source_file, section_title, etc.)
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from icca.document_parser.parser import ParsedDocument, ParsedElement


@dataclass
class TextChunk:
    """
    Structured text unit $t_{i,j}$ representing the j-th section or paragraph of document $d_i$.
    """
    chunk_id: str
    doc_id: str
    text: str
    section_num: int
    paragraph_num: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return asdict(self)


class DocumentChunker:
    """
    Segments parsed documents into coherent semantic chunks.
    
    Adheres to Algorithm 1 (Lines 36-38):
    - If document length <= threshold, store as a single text unit $t_{i,1} = d_i$.
    - If document length > threshold, split into semantic sections and paragraphs.
    - Tags customs-specific entities (HS codes, regulatory terms, procedures).
    """

    # Precise HS code patterns:
    # 1. With prefix: HS 8429, Heading 8471, Tariff 8479, Code 8504
    # 2. Dot notation: 8429.51, 8471.30.10, 8504.40.90
    # 3. Known tariff chapters: 84xx or 85xx when standing alone
    HS_PREFIX_REGEX = re.compile(r"\b(?:HS|Heading|Subheading|Tariff|Code)\s*[:#-]?\s*(\d{4}(?:\.\d{2}(?:\.\d{2})?)?)\b", re.IGNORECASE)
    HS_DOT_REGEX = re.compile(r"\b(\d{4}\.\d{2}(?:\.\d{2})?)\b")
    HS_CHAPTER_REGEX = re.compile(r"\b((?:84|85|28|29)\d{2})\b")

    # Common years to ignore from chapter regex
    IGNORE_YEARS = {"1962", "1975", "1985", "1986", "1989", "1992", "2000", "2005", "2007", "2010", "2016", "2020", "2023", "2024", "2025", "2026"}

    REGULATORY_TERMS = [
        "Bill of Entry", "Shipping Bill", "Harmonized System", "HS Code",
        "Basic Customs Duty", "BCD", "IGST", "AIDC", "SWS",
        "SCOMET", "Dual-Use", "DGFT", "End-User Certificate",
        "BIS", "CRO", "CRS", "Chartered Engineer", "MSDS", "Hazmat",
        "IMDG", "ICAO", "Dangerous Goods", "AEO", "Risk Management System",
        "RMS", "EDI", "ICEGATE", "Out of Charge", "OOC", "First Check", "Second Check",
        "Valuation", "Bonded Warehouse", "CPCB", "SPCB", "EPCG", "MOOWR", "CRCL",
        "Incoterms", "Certificate of Origin"
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_threshold: int = 600
    ):
        """
        Initialize chunker with size and overlap limits.
        
        Args:
            chunk_size: Target maximum word / character length for a paragraph chunk.
            chunk_overlap: Overlap length when splitting oversized paragraphs.
            length_threshold: If full document character count is below this threshold,
                              it is kept as a single short document unit $t_{i,1} = d_i$.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_threshold = length_threshold

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Segment a parsed document into structured text units $t_{i,j}$.
        
        Args:
            parsed_doc: ParsedDocument instance from DocumentParser.
            
        Returns:
            List of TextChunk instances with metadata.
        """
        full_text = parsed_doc.get_full_text().strip()
        doc_id = parsed_doc.doc_id
        doc_type = parsed_doc.doc_type

        # Case 1: Empty document
        if not full_text:
            return []

        # Case 2: Short document (Algorithm 1: Length <= Threshold -> t_{i,1} = d_i)
        if len(full_text) <= self.length_threshold:
            tags = self.extract_tags(full_text)
            chunk = TextChunk(
                chunk_id=f"{doc_id}_c0",
                doc_id=doc_id,
                text=full_text,
                section_num=1,
                paragraph_num=1,
                token_count=self._estimate_tokens(full_text),
                metadata={
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "source_file": parsed_doc.file_path,
                    "chunk_type": "single_document",
                    "tags": tags,
                    "is_short_doc": True,
                    **parsed_doc.metadata
                }
            )
            return [chunk]

        # Case 3: Long document (Algorithm 1: Length > Threshold -> SplitTextBySections)
        chunks: List[TextChunk] = []
        chunk_index = 0
        current_section_num = 1
        current_section_title = ""
        current_para_num = 1

        # If document has structured elements from parser
        if parsed_doc.elements:
            for elem in parsed_doc.elements:
                content = elem.content.strip()
                if not content or content == "---":
                    continue

                if elem.element_type == "title":
                    current_section_num += 1
                    current_para_num = 1
                    current_section_title = content.lstrip("#").strip()
                    # Store title context and continue (it will prepend to subsequent paragraphs)
                    continue

                if elem.element_type == "table":
                    # Tables are treated as structured units
                    table_chunks = self._chunk_table(content, doc_id, current_section_num, current_para_num)
                    for t_text in table_chunks:
                        chunk_id = f"{doc_id}_c{chunk_index}"
                        tagged_text = f"[{current_section_title}]\n{t_text}" if current_section_title else t_text
                        tags = self.extract_tags(tagged_text)
                        chunks.append(TextChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            text=tagged_text,
                            section_num=current_section_num,
                            paragraph_num=current_para_num,
                            token_count=self._estimate_tokens(tagged_text),
                            metadata={
                                "doc_id": doc_id,
                                "doc_type": doc_type,
                                "source_file": parsed_doc.file_path,
                                "chunk_type": "table",
                                "section_title": current_section_title,
                                "tags": tags,
                                **elem.metadata
                            }
                        ))
                        chunk_index += 1
                        current_para_num += 1
                    continue

                # Paragraph or OCR element
                para_segments = self._split_paragraph(content)
                for seg in para_segments:
                    chunk_id = f"{doc_id}_c{chunk_index}"
                    contextual_text = f"[{current_section_title}]\n{seg}" if current_section_title else seg
                    tags = self.extract_tags(contextual_text)
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=contextual_text,
                        section_num=current_section_num,
                        paragraph_num=current_para_num,
                        token_count=self._estimate_tokens(contextual_text),
                        metadata={
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "source_file": parsed_doc.file_path,
                            "chunk_type": elem.element_type,
                            "section_title": current_section_title,
                            "tags": tags,
                            **elem.metadata
                        }
                    ))
                    chunk_index += 1
                    current_para_num += 1
        else:
            # Fallback text segmentation
            return self.chunk_text(full_text, doc_id=doc_id, doc_type=doc_type, metadata=parsed_doc.metadata)

        return chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc",
        doc_type: str = "txt",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """
        Segment raw text into structured text units.
        """
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.length_threshold:
            tags = self.extract_tags(text)
            return [TextChunk(
                chunk_id=f"{doc_id}_c0",
                doc_id=doc_id,
                text=text,
                section_num=1,
                paragraph_num=1,
                token_count=self._estimate_tokens(text),
                metadata={
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "tags": tags,
                    "is_short_doc": True,
                    **metadata
                }
            )]

        # Section-based splitting
        section_pattern = re.compile(
            r"(?=(?:^|\n)(?:#{1,4}\s+|SECTION\s+[0-9IVXLCDM]+|CHAPTER\s+[0-9IVXLCDM]+|PART\s+[0-9IVXLCDM]+|STEP\s+[0-9]+|STAGE\s+[0-9]+))",
            re.IGNORECASE
        )
        raw_sections = [s.strip() for s in section_pattern.split(text) if s.strip()]

        chunks: List[TextChunk] = []
        chunk_index = 0

        for sec_idx, section in enumerate(raw_sections, start=1):
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
            sec_title = ""
            for para_idx, para in enumerate(paragraphs, start=1):
                if para.startswith("#") or re.match(r"^(SECTION|CHAPTER|STAGE|STEP)\s+[0-9IVXLCDM]+", para, re.IGNORECASE):
                    sec_title = para.splitlines()[0].lstrip("#").strip()
                    lines = para.splitlines()[1:]
                    para = "\n".join(lines).strip()
                    if not para:
                        continue

                segments = self._split_paragraph(para)
                for seg in segments:
                    chunk_id = f"{doc_id}_c{chunk_index}"
                    chunk_text_with_ctx = f"[{sec_title}]\n{seg}" if sec_title else seg
                    tags = self.extract_tags(chunk_text_with_ctx)
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=chunk_text_with_ctx,
                        section_num=sec_idx,
                        paragraph_num=para_idx,
                        token_count=self._estimate_tokens(chunk_text_with_ctx),
                        metadata={
                            "doc_id": doc_id,
                            "doc_type": doc_type,
                            "chunk_type": "paragraph",
                            "section_title": sec_title,
                            "tags": tags,
                            **metadata
                        }
                    ))
                    chunk_index += 1

        return chunks

    def _split_paragraph(self, text: str) -> List[str]:
        """
        Split a paragraph into segments if it exceeds chunk_size characters.
        Uses sliding window with chunk_overlap, respecting sentence boundaries.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Sentence split
        sentences = re.split(r"(?<=[.!?])\s+", text)
        segments: List[str] = []
        current_segment: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current_segment:
                seg_text = " ".join(current_segment).strip()
                segments.append(seg_text)

                # Overlap: keep the last sentence(s) up to chunk_overlap length
                overlap_segment: List[str] = []
                overlap_len = 0
                for s in reversed(current_segment):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_segment.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_segment = overlap_segment + [sent]
                current_len = sum(len(s) for s in current_segment) + len(current_segment)
            else:
                current_segment.append(sent)
                current_len += sent_len + 1

        if current_segment:
            seg_text = " ".join(current_segment).strip()
            if seg_text:
                segments.append(seg_text)

        return segments if segments else [text]

    def _chunk_table(self, table_md: str, doc_id: str, section_num: int, para_num: int) -> List[str]:
        """
        Split large Markdown tables while preserving header rows.
        """
        lines = [line.strip() for line in table_md.split("\n") if line.strip()]
        if len(lines) <= 2:
            return [table_md]

        # If total character length is manageable, return as single chunk
        if len(table_md) <= self.chunk_size * 2:
            return [table_md]

        header = lines[0]
        separator = lines[1] if len(lines) > 1 and "---" in lines[1] else "|---|"
        data_rows = lines[2:] if "---" in lines[1] else lines[1:]

        chunks = []
        rows_per_chunk = max(3, self.chunk_size // 150)
        for i in range(0, len(data_rows), rows_per_chunk):
            sub_rows = data_rows[i:i + rows_per_chunk]
            chunk_table = "\n".join([header, separator] + sub_rows)
            chunks.append(chunk_table)

        return chunks if chunks else [table_md]

    def extract_tags(self, text: str) -> List[str]:
        """
        Extract customs-relevant tags from text, such as HS codes and regulatory keywords.
        """
        tags = set()

        # 1. Prefixed HS codes (e.g. HS 8429, Heading 8471)
        for match in self.HS_PREFIX_REGEX.finditer(text):
            code = match.group(1).strip()
            tags.add(f"HS_{code}")

        # 2. Dot-notated tariff codes (e.g. 8429.51, 8471.30.10)
        for match in self.HS_DOT_REGEX.finditer(text):
            code = match.group(1).strip()
            tags.add(f"HS_{code}")

        # 3. Known tariff chapters (e.g. 8429, 8471, 8504) excluding years
        for match in self.HS_CHAPTER_REGEX.finditer(text):
            code = match.group(1).strip()
            if code not in self.IGNORE_YEARS:
                tags.add(f"HS_{code}")

        # 4. Extract regulatory keywords
        text_lower = text.lower()
        for term in self.REGULATORY_TERMS:
            # Word boundary matching for short acronyms like BCD, SWS, OOC, RMS, EDI
            if len(term) <= 4 and term.isupper():
                if re.search(r"\b" + re.escape(term) + r"\b", text):
                    tags.add(term)
            else:
                if term.lower() in text_lower:
                    tags.add(term)

        return sorted(list(tags))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count based on words and punctuation."""
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))
