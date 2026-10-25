"""Turn a file on disk into a cleaned `ParsedDocument` with metadata."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ingestion.cleaner import clean_blocks
from ingestion.html_parser import HtmlParser
from ingestion.metadata import (
    classify_doc_type,
    extract_addendum_number,
    extract_document_date,
    file_sha256,
    pdf_metadata_date,
)
from ingestion.pdf_parser import PdfParser
from schemas import BlockType, DocType, DocumentMetadata, ParsedDocument

log = logging.getLogger(__name__)

PDF_EXT = {".pdf"}
HTML_EXT = {".html", ".htm"}


class UnsupportedFileError(ValueError):
    pass


class DocumentLoader:
    def __init__(self, pdf_parser: Optional[PdfParser] = None, html_parser: Optional[HtmlParser] = None):
        self._pdf = pdf_parser
        self.html = html_parser or HtmlParser()

    @property
    def pdf(self) -> PdfParser:
        if self._pdf is None:
            self._pdf = PdfParser()
        return self._pdf

    def load(
        self, path: Path, bid_id: str, content_hash: Optional[str] = None, ocr: Optional[bool] = None
    ) -> ParsedDocument:
        """`ocr` overrides the configured PDF OCR setting for this document (None = default)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {path.name}")
        ext = path.suffix.lower()
        content_hash = content_hash or file_sha256(path)

        warnings: list[str] = []
        empty_pages: list[int] = []
        pdf_date: Optional[str] = None
        ocr_used: Optional[bool] = None
        if ext in PDF_EXT:
            res = self.pdf.parse(path, ocr=ocr)
            ocr_used = res.ocr
            blocks, num_pages, parser = res.blocks, res.num_pages, res.parser
            warnings += res.warnings
            empty_pages = res.empty_pages
            pdf_date = pdf_metadata_date(res.pdf_creation_date)
            title = None
        elif ext in HTML_EXT:
            blocks, title, html_warnings = self.html.parse_with_info(path)
            warnings += html_warnings
            num_pages, parser = None, "beautifulsoup"
        else:
            raise UnsupportedFileError(f"Unsupported file type: {ext}")

        blocks = clean_blocks(blocks, num_pages)
        head_text = "\n".join(b.text for b in blocks[:25])
        doc_type = classify_doc_type(path.name, head_text)
        content_date = extract_document_date(head_text)
        if title is None:
            title = next((b.text for b in blocks[:10] if b.type == BlockType.HEADING), None)

        md = DocumentMetadata(
            bid_id=bid_id,
            file_name=path.name,
            source_path=str(path.resolve()),
            doc_type=doc_type,
            addendum_number=(
                extract_addendum_number(path.name, head_text) if doc_type == DocType.ADDENDUM else None
            ),
            document_date=content_date or pdf_date,
            document_date_source="content" if content_date else ("pdf_metadata" if pdf_date else None),
            title=title,
            num_pages=num_pages,
            content_hash=content_hash,
            parser=parser,
            ocr_used=ocr_used,
        )
        doc = ParsedDocument(metadata=md, blocks=blocks, warnings=warnings, empty_pages=empty_pages)
        if doc.is_empty:
            doc.warnings.append("No extractable text (empty, scanned without OCR, or unreadable)")
        return doc
