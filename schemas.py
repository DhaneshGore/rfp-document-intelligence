"""Typed data structures shared by ingestion, search, services, API and (later) agents."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    BID_PAGE = "bid_page"
    RFP = "rfp"
    ADDENDUM = "addendum"
    SPECS = "specs"
    AFFIDAVIT = "affidavit"
    OTHER = "other"


class BlockType(str, Enum):
    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"


class Block(BaseModel):
    """A structural unit produced by a parser (paragraph, heading, table...)."""

    type: BlockType
    text: str
    page: Optional[int] = None  # 1-based physical page; None for HTML
    level: int = 1  # heading level
    # Tables only: header row and body rows (already rendered as "a | b | c" strings)
    table_header: Optional[str] = None
    table_rows: Optional[list[str]] = None


class DocumentMetadata(BaseModel):
    bid_id: str
    file_name: str
    source_path: str
    doc_type: DocType = DocType.OTHER
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None  # ISO yyyy-mm-dd when parseable
    document_date_source: Optional[str] = None  # "content" | "pdf_metadata" | ...
    title: Optional[str] = None
    num_pages: Optional[int] = None
    content_hash: str = ""
    parser: str = ""
    ocr_used: Optional[bool] = None  # PDFs only: whether OCR ran


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    blocks: list[Block] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    empty_pages: list[int] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(b.text.strip() for b in self.blocks)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str  # text that is indexed (includes section context prefix)
    bid_id: str
    file_name: str
    source_path: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    page_number: Optional[int] = None  # first page the chunk comes from
    page_end: Optional[int] = None  # last page (== page_number unless spanning)
    document_date: Optional[str] = None
    section: Optional[str] = None
    chunk_type: str = "text"  # text | table
    chunk_index: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata for the vector store (ChromaDB rejects None values)."""
        md = self.model_dump(exclude={"text"})
        md["doc_type"] = self.doc_type.value
        return {k: v for k, v in md.items() if v is not None}

    @classmethod
    def from_store(cls, chunk_id: str, text: str, md: dict[str, Any]) -> "Chunk":
        data = dict(md)
        data["chunk_id"] = chunk_id
        data["text"] = text
        return cls.model_validate(data)


class Citation(BaseModel):
    file_name: str
    page_number: Optional[int] = None
    page_end: Optional[int] = None
    bid_id: str
    doc_type: str
    chunk_id: str
    source_path: str
    section: Optional[str] = None  # heading path; locates page-less (HTML) passages

    @property
    def label(self) -> str:
        if self.page_number is None:
            if not self.section:
                return self.file_name
            heading = self.section.split(" > ")[-1]
            heading = heading if len(heading) <= 60 else heading[:57].rstrip() + "..."
            return f'{self.file_name}, section "{heading}"'
        if self.page_end and self.page_end != self.page_number:
            return f"{self.file_name}, pp. {self.page_number}-{self.page_end}"
        return f"{self.file_name}, p. {self.page_number}"


class SearchResult(BaseModel):
    rank: int
    score: float  # final score used for ordering (rerank > rrf > raw)
    chunk_id: str
    text: str
    bid_id: str
    file_name: str
    doc_type: str
    page_number: Optional[int] = None
    page_end: Optional[int] = None
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None
    section: Optional[str] = None
    source_path: str
    citation: str  # human-readable "file, p. N"
    scores: dict[str, float] = Field(default_factory=dict)  # dense/bm25/rrf/rerank
    ranks: dict[str, int] = Field(default_factory=dict)  # per-retriever ranks


class SearchFilters(BaseModel):
    bid_id: Optional[str] = None
    doc_type: Optional[str] = None
    addendum_number: Optional[int] = None
    file_name: Optional[str] = None

    def active(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v not in (None, "")}


class FileIndexStatus(BaseModel):
    source_path: str
    bid_id: str
    file_name: str
    status: str  # indexed | skipped_unchanged | skipped_duplicate | failed | removed
    doc_type: Optional[str] = None
    chunks: int = 0
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    seconds: float = 0.0


class IndexReport(BaseModel):
    files: list[FileIndexStatus] = Field(default_factory=list)
    total_chunks_added: int = 0
    seconds: float = 0.0

    @property
    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.files:
            out[f.status] = out.get(f.status, 0) + 1
        return out


class LLMUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None  # USD as reported by OpenRouter (0 for free models)


# OpenAI-style "length" (OpenRouter's normalised value) and native provider spellings.
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "model_length"})


class LLMResponse(BaseModel):
    text: str
    model: str
    latency_s: float
    usage: LLMUsage = Field(default_factory=LLMUsage)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: Optional[str] = None  # provider's reason; "length" = stopped at max_tokens

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() in TRUNCATION_REASONS


class AnswerCitation(BaseModel):
    ref: int  # [n] marker used in the answer
    citation: str
    file_name: str
    page_number: Optional[int] = None
    page_end: Optional[int] = None
    bid_id: str
    doc_type: str
    chunk_id: str
    text: str


class Answer(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[AnswerCitation] = Field(default_factory=list)
    evidence: list[SearchResult] = Field(default_factory=list)
    model: Optional[str] = None
    latency_s: Optional[float] = None
    usage: Optional[LLMUsage] = None
    invalid_citation_refs: list[int] = Field(default_factory=list)
    # numbers/dates/times in the answer that its cited passages do not contain
    ungrounded_values: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    # verified: cited and every value grounded | unverified: answer with citation or grounding
    # problems (see error) | incomplete: stopped at the token limit | not_found | error
    status: str = "verified"
    finish_reason: Optional[str] = None
