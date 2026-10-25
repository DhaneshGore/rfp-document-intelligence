"""Typed input / output contracts for every agent (all agent-to-agent messages).

Each model has `summary()` - the small, text-free view written to the trace (evidence is
referenced by chunk id / file / page, never copied).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from extraction.schemas import AddendumChange, BidExtraction, FieldResult, ValidationSummary
from schemas import Answer, SearchFilters, SearchResult


def _refs(passages: list[SearchResult]) -> list[dict[str, Any]]:
    return [{"chunk_id": p.chunk_id, "file": p.file_name, "page": p.page_number, "doc_type": p.doc_type}
            for p in passages]


# ----------------------------------------------------------------------- ingestion
class IngestionInput(BaseModel):
    folder: str
    bid_id: Optional[str] = None  # default: folder name

    def summary(self) -> dict:
        return {"folder": self.folder, "bid_id": self.bid_id}


class IngestionOutput(BaseModel):
    bid_id: str
    files: dict[str, int] = Field(default_factory=dict)  # status -> count
    chunks_added: int = 0
    failed_files: list[str] = Field(default_factory=list)

    def summary(self) -> dict:
        return self.model_dump()


# ----------------------------------------------------------------------- retrieval
class RetrievalInput(BaseModel):
    bid_id: Optional[str] = None
    label: str  # e.g. "identity", "identity/retry", "reconcile/addendum", "question"
    fields: list[str] = Field(default_factory=list)  # field-evidence mode
    scale: int = 1
    extra_queries: bool = False
    scope: Optional[Literal["original", "addendum"]] = None
    question: Optional[str] = None  # question mode
    filters: Optional[SearchFilters] = None
    mode: Optional[str] = None
    top_k: Optional[int] = None

    def summary(self) -> dict:
        return self.model_dump(exclude_none=True, exclude={"filters"}) | (
            {"filters": self.filters.active()} if self.filters else {})


class RetrievalOutput(BaseModel):
    label: str
    bid_id: Optional[str] = None
    passages: list[SearchResult] = Field(default_factory=list)
    ranks: dict[str, dict[str, int]] = Field(default_factory=dict)  # field -> {chunk_id: best rank}
    dropped_foreign: int = 0

    def summary(self) -> dict:
        return {"label": self.label, "passages": _refs(self.passages), "dropped_foreign": self.dropped_foreign}


# ----------------------------------------------------------------------- extraction
class ExtractionInput(BaseModel):
    bid_id: str
    label: str
    fields: list[str]
    evidence: RetrievalOutput
    attempt: int = 1
    scope: Optional[Literal["original", "addendum"]] = None
    note: Optional[str] = None  # focused retries: why the field is asked again

    def summary(self) -> dict:
        return {"label": self.label, "fields": self.fields, "attempt": self.attempt, "scope": self.scope,
                "passages": len(self.evidence.passages)}


class ExtractionOutput(BaseModel):
    label: str
    results: dict[str, FieldResult]
    notes: dict[str, str] = Field(default_factory=dict)
    model: Optional[str] = None
    llm_called: bool = False
    llm_calls: int = 0  # > 1 when a truncated group was split
    change_types: dict[str, str] = Field(default_factory=dict)  # addendum scope only

    def summary(self) -> dict:
        return {"label": self.label, "llm_calls": self.llm_calls, "model": self.model,
                "statuses": {k: v.status for k, v in self.results.items()}}


# ----------------------------------------------------------------------- validation (critic)
Decision = Literal["accept", "retry", "not_found", "give_up"]


class FieldVerdict(BaseModel):
    field: str
    decision: Decision
    reasons: list[str] = Field(default_factory=list)


class ValidationInput(BaseModel):
    bid_id: str
    fields: dict[str, FieldResult]
    shown: dict[str, set[str]]  # field -> chunk ids of the evidence the model saw for it
    max_attempts: int
    retry_not_found: bool = False
    final: bool = False  # last pass: no retries can follow

    def summary(self) -> dict:
        return {"fields": len(self.fields), "max_attempts": self.max_attempts, "final": self.final}


class ValidationOutput(BaseModel):
    fields: dict[str, FieldResult]  # possibly downgraded by the critic
    verdicts: dict[str, FieldVerdict]
    retry: list[str] = Field(default_factory=list)
    summary_counts: ValidationSummary
    warnings: list[str] = Field(default_factory=list)
    downgraded: list[str] = Field(default_factory=list)

    def summary(self) -> dict:
        return {"retry": self.retry, "downgraded": self.downgraded, "warnings": self.warnings,
                "counts": self.summary_counts.model_dump()}


# ----------------------------------------------------------------------- reconciliation
class ReconciliationRequest(BaseModel):
    """A scoped extraction the reconciliation agent needs (run on the Extraction agent)."""
    scope: Literal["original", "addendum"]
    group: str
    fields: list[str]


class ReconciliationInput(BaseModel):
    bid_id: str
    fields: dict[str, FieldResult]
    latest: dict[str, FieldResult]  # addendum-scope extraction
    original: dict[str, FieldResult]  # original-documents extraction
    change_types: dict[str, str] = Field(default_factory=dict)  # model's change label per field
    addendum_texts: dict[str, str] = Field(default_factory=dict)  # chunk id -> addendum passage text

    def summary(self) -> dict:
        return {"fields": sorted(self.latest), "change_types": self.change_types}


class ReconciliationOutput(BaseModel):
    fields: dict[str, FieldResult]
    changes: list[AddendumChange] = Field(default_factory=list)

    def summary(self) -> dict:
        return {"changes": [{"field": c.field, "addendum_number": c.addendum_number} for c in self.changes]}


# ----------------------------------------------------------------------- Q&A / report
class QAInput(BaseModel):
    question: str
    evidence: RetrievalOutput
    model: Optional[str] = None

    def summary(self) -> dict:
        return {"question": self.question[:200], "passages": len(self.evidence.passages)}


class QAOutput(BaseModel):
    answer: Answer

    def summary(self) -> dict:
        a = self.answer
        return {"status": a.status, "found": a.found, "citations": [c.chunk_id for c in a.citations],
                "model": a.model, "error": (a.error or "")[:300] or None}


class ReportInput(BaseModel):
    bid_id: str
    fields: dict[str, FieldResult]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationSummary
    model: Optional[str] = None
    llm_calls: int = 0

    def summary(self) -> dict:
        return {"bid_id": self.bid_id, "validation": self.validation.model_dump()}


class ReportOutput(BaseModel):
    record: BidExtraction
    markdown: str

    def summary(self) -> dict:
        return {"bid_id": self.record.bid_id, "validation": self.record.validation.model_dump(),
                "report_chars": len(self.markdown)}
