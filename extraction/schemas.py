"""Output models for structured extraction (assignment Section 8.1).

One `BidExtraction` per bid; `fields` always holds all 20 fields. A value that is absent
or unsupported is `null`, and `notes` says why. `confidence` is computed from evidence
(see `extraction.service.score_confidence`), never taken from the model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

FieldStatus = Literal["verified", "unverified", "not_found", "error"]
FieldValue = Optional[Union[str, list[str]]]


class FieldSource(BaseModel):
    file: str
    page: Optional[int] = None  # None for HTML pages (no pagination)
    page_end: Optional[int] = None
    chunk_id: str
    citation: str  # human-readable label, e.g. "RFP.pdf, p. 3"
    doc_type: Optional[str] = None
    addendum_number: Optional[int] = None


class FieldResult(BaseModel):
    value: FieldValue = None
    sources: list[FieldSource] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    status: FieldStatus = "not_found"
    attempts: int = 1  # extraction attempts (1 + retries)


class AddendumChange(BaseModel):
    """A field whose value an addendum changed. Recorded only when both the original value
    (cited from non-addendum documents) and the latest addendum value (cited from an
    addendum) passed evidence validation, and they differ."""
    field: str
    value: FieldValue
    addendum_number: Optional[int] = None
    sources: list[FieldSource] = Field(default_factory=list)
    previous_value: FieldValue = None
    previous_sources: list[FieldSource] = Field(default_factory=list)
    detected_by: str = "reconciliation"
    notes: str = ""


class ValidationSummary(BaseModel):
    passed: int = 0  # verified
    failed: int = 0  # unverified + error
    not_found: int = 0
    unverified: int = 0
    error: int = 0


class BidExtraction(BaseModel):
    bid_id: str
    fields: dict[str, FieldResult]
    addendum_changes: list[AddendumChange] = Field(default_factory=list)
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
    model: Optional[str] = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    llm_calls: int = 0
