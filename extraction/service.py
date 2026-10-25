"""Grouped, evidence-validated extraction of the 20 Section 8 fields for one bid.

Flow per bid:
  1. Extraction - one LLM call per field group (5 groups). Each field's queries go through
     `RetrievalService.search`, always filtered to the bid (addendum-sensitive fields also
     search `doc_type=addendum`); results are re-checked to belong to the bid,
     de-duplicated and capped per group. The model gets numbered passages
     (`qa_service.format_evidence`) and must return JSON {field: {"value", "refs", "notes"}}.
  2. Validation per field (`validate_value`): refs must point at passages that exist;
     numbers must be grounded in the cited passages (`qa_service.grounding_details`);
     identifiers / emails / phones must appear verbatim; free text must mostly come from
     the cited passages; dates must parse.
  3. Retries (bounded, `max_retries` rounds): `error` fields are retried; `unverified`
     fields (and `not_found` fields if `retry_not_found`) only when a wider search finds
     passages that field has not seen yet. A retry result replaces the earlier one only if
     it is better (verified > unverified > not_found > error, then confidence).
  4. Addendum reconciliation for addendum-sensitive fields, if the bid has addendum
     passages: per field group, one addendum-only call (value only if an addendum really
     changes the requirement, plus a "change" label); then original-document calls only for
     fields with a validated substantive addendum value. A change is recorded only when both
     values validate, the change is substantive (label + change wording in the cited
     addendum passage) and the values differ in substance (`apply_reconciliation`).
  Identical prompts over identical evidence are never sent twice (`_requests`); a group whose
  JSON hits the output-token limit is split (`_split`).
  Status: verified (cited + every check passed), unverified (cited but a check failed, or
  fabricated refs -> value nulled), not_found (model returned null), error (LLM failure,
  truncated or invalid JSON, field omitted).

Confidence (`score_confidence`) is evidence-based and ignores anything the model says:
    found:     0.35*citation + 0.35*support + 0.15*format + 0.15*retrieval
               citation  1.0 all refs valid | 0.5 some refs fabricated | 0 none valid
               support   share of the value's words/identifiers found in the cited passages,
                         0 if any number is not in them
               format    1 if the kind's format rule passes (date parses, contact has
                         email/phone, identifier has a digit), else 0
               retrieval 1.0 if a cited passage was the field's own rank-1 search hit,
                         decaying with rank (1/(1+(rank-1)/3)); 0.25 if it was only found
                         by another field's query
               unverified values are capped at 0.5
    not_found: 0.6 * min(1, passages retrieved for the field / per_query) - how thoroughly
               the documents were searched, not certainty that the value is absent
    error:     0.0
Retrieval relevance scores stay in the evidence; they are not copied into confidence.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from extraction.fields import BY_NAME, FIELD_NAMES, FIELDS, GROUPS, FieldSpec, fields_in
from extraction.schemas import AddendumChange, BidExtraction, FieldResult, FieldSource, ValidationSummary
from extraction.textnorm import normalize_text, states_positively
from extraction.trace import ExtractionTrace, evidence_refs
from ingestion.metadata import parse_date
from schemas import SearchFilters, SearchResult
from search.keyword_search import STOPWORDS
from services.llm_service import LLMError, redact
from services.qa_service import (EVIDENCE_PASSAGE_CHARS, _numbers, focus_patterns, format_evidence, grounding_details,
                                 strip_reasoning)

log = logging.getLogger(__name__)

NOT_FOUND_NOTE = "Not found in documents"
# Single-field call that hit the output limit (reasoning models spend the budget thinking):
# asked once more over fewer passages with this note.
COMPACT_NOTE = ("Output budget is small: decide quickly, do not deliberate at length, and reply with the JSON "
                "object only.")
COMPACT_PASSAGES = 4
_NULL_STRINGS = {"", "null", "not found", "not found in documents", "n/a", "na", "unknown", "not stated"}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?:\s*(?:x|ext\.?)\s*\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")
# Minimum share of a value's content words that must occur in its cited passages.
SUPPORT_THRESHOLD = {"id": 1.0, "date": 0.5, "contact": 0.5, "text": 0.6, "summary": 0.4}
_STATUS_RANK = {"verified": 3, "unverified": 2, "not_found": 1, "error": 0}
ADDENDUM = "addendum"

SYSTEM_PROMPT = """You extract structured fields from bid / RFP documents using ONLY the numbered evidence passages.

Return ONE JSON object and nothing else (no markdown, no reasoning). Its keys are exactly the
requested field names. Each key maps to:
  {"value": <string, list of strings, or null>, "refs": [<passage numbers>], "notes": "<short note>"}

Rules:
1. Copy identifiers, dates, times, time zones, amounts, names, emails and phone numbers exactly as
   written in the passages. Do not reformat or convert them.
2. "refs" lists the numbers of the passages that contain the value. Never cite a passage that does
   not contain it, and never cite a number that is not in the list.
3. If the passages do not state the field, use "value": null, "refs": [] and
   "notes": "Not found in documents". Do not guess or use outside knowledge.
4. Passages are labelled with document type and addendum number. Addendums override the base RFP:
   use the latest addendum's value, cite it, and mention the original value in "notes".
5. For fields that can have several values, return a list of strings.
6. Be brief: "notes" at most 20 words; values as short as the field allows (Bid Summary 3-6 sentences)."""

SCOPE_NOTES = {
    "original": "Scope: these passages come from the ORIGINAL bid documents only (no addendums). Give each "
                "field's value as originally stated; null if these passages do not state it.",
    ADDENDUM: "Scope: these passages come from ADDENDUMS only. Give a field's value ONLY if an addendum "
              "explicitly changes the requirement (extends, replaces, adds, removes or materially alters it); "
              "give the value as set by the latest such addendum. If the addendum passages only clarify, answer "
              "a question without changing the requirement, reword or restate it, use \"value\": null.\n"
              "Each field object must also have \"change\": one of \"changed\", \"extended\", \"replaced\", "
              "\"added\", \"removed\" (a real change), or \"clarification\" or \"none\" (no change).",
}

# A change is recorded only when the model labels it substantive AND the cited addendum
# passage itself contains change wording (evidence, not just the model's say-so).
SUBSTANTIVE_CHANGES = frozenset({"changed", "extended", "replaced", "added", "removed"})
_CHANGE_MARKER_RE = re.compile(
    r"\b(extend\w*|chang\w*|revis\w*|replac\w*|amend\w*|modif\w*|add(?:ed|s|ing)?|delet\w*|remov\w*|new|instead|"
    r"in lieu|no longer|postpon\w*|cancel\w*|substitut\w*|increas\w*|decreas\w*|supersed\w*|"
    r"now (?:be|is|are|reads?))\b", re.I)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+|\n+|\s\|\s")


# ----------------------------------------------------------------------- helpers
def _content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in STOPWORDS]


def _norm(text: str) -> str:
    return normalize_text(text)  # lower case, Unicode spaces / dashes folded


def _as_items(value) -> list[str]:
    return value if isinstance(value, list) else [value]


def same_value(a, b) -> bool:
    """Equal after case, whitespace and punctuation normalisation (lists: as sets)."""
    def norm(x: str) -> str:
        return " ".join(re.sub(r"[^\w:/@.-]+", " ", x.lower()).split()).strip(" .")
    return {norm(x) for x in _as_items(a)} == {norm(x) for x in _as_items(b)}


def substantively_same(a, b) -> bool:
    """True when two values say the same thing in different words / formats: equal after
    normalisation, or the same date and time ("27-JUN-2024 14:00" == "June 27, 2024 2:00 PM"),
    or the same numbers with largely the same words (rewording)."""
    if same_value(a, b):
        return True
    ta, tb = " ".join(_as_items(a)), " ".join(_as_items(b))
    da, db = parse_date(ta), parse_date(tb)
    if da or db:
        if da != db:
            return False
        times = lambda t: {(int(h) % 12, int(m)) for h, m in _TIME_RE.findall(t)}  # noqa: E731
        return times(ta) == times(tb) or not (times(ta) and times(tb))
    wa, wb = set(_content_words(ta)), set(_content_words(tb))
    overlap = len(wa & wb) / len(wa | wb) if wa | wb else 1.0
    return _numbers(ta) == _numbers(tb) and overlap >= 0.8


def normalize_value(raw) -> Optional[object]:
    """Model value -> str | list[str] | None (null-like strings become None)."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if x is not None and str(x).strip()]
        items = [x for x in items if x.lower().rstrip(".") not in _NULL_STRINGS]
        return items or None
    text = str(raw).strip()
    return None if text.lower().rstrip(".") in _NULL_STRINGS else text


def parse_json_object(text: str) -> dict:
    """First top-level JSON object in a model reply (tolerates ``` fences and <think>)."""
    text = strip_reasoning(text or "")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def field_patterns(specs: list[FieldSpec]) -> list[re.Pattern]:
    """What makes a line of a long passage relevant to these fields: their cue and coverage
    regexes plus the words of their search queries and names."""
    pats = [re.compile(s.cue, re.I) for s in specs if s.cue]
    pats += [re.compile(p, re.I) for s in specs for _, p in s.coverage]
    return pats + focus_patterns(*(q for s in specs for q in s.queries), *(s.name for s in specs))


def build_extraction_messages(specs: list[FieldSpec], evidence: list[SearchResult],
                              passage_chars: int = EVIDENCE_PASSAGE_CHARS, scope: Optional[str] = None,
                              note: Optional[str] = None) -> list[dict[str, str]]:
    fields = "\n".join(f'- "{s.name}": {s.description}' + (" (may be a list)" if s.multi else "")
                       + (f" {s.hint}" if s.hint else "") for s in specs)
    scope_line = (f"{SCOPE_NOTES[scope]}\n\n" if scope else "") + (f"{note}\n\n" if note else "")
    user = (f"{scope_line}Fields to extract:\n{fields}\n\nEvidence passages:\n\n"
            f"{format_evidence(evidence, passage_chars, patterns=field_patterns(specs))}\n\n"
            f"Return the JSON object with exactly these keys: {json.dumps([s.name for s in specs])}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ----------------------------------------------------------------------- validation
def validate_value(spec: FieldSpec, value, refs: list[int], cited: list[SearchResult],
                   evidence: list[SearchResult]) -> tuple[bool, float, bool, bool, list[str]]:
    """Check a cited value against its passages.

    Returns (ok, support, numbers_ok, format_ok, problems).
    """
    items = _as_items(value)
    flat = " ".join(" ".join(i.split()) for i in items)
    cited_text = "\n".join(r.text for r in cited)
    cited_norm = _norm(cited_text)
    problems: list[str] = []

    g = grounding_details(f"{flat} [{', '.join(map(str, refs))}]", cited, evidence, question=spec.description)
    numbers_ok = not g["not_in_cited"]
    if not numbers_ok:
        problems.append(f"numbers not in the cited passages: {', '.join(g['not_in_cited'])}")

    format_ok = True
    if spec.kind == "id":
        found = [i for i in items if _norm(i) in cited_norm]
        support = len(found) / len(items)
        if support < 1:
            problems.append(f"not found verbatim in the cited passages: {[i for i in items if i not in found]}")
        format_ok = all(any(c.isdigit() for c in i) for i in items)
        if not format_ok:
            problems.append("identifier without any digit")
    else:
        words = _content_words(flat)
        cited_words = set(_content_words(cited_text))
        support = (sum(w in cited_words for w in words) / len(words)) if words else (1.0 if numbers_ok else 0.0)
        if spec.kind == "date":
            format_ok = parse_date(flat) is not None
            if not format_ok:
                problems.append("no parseable date")
        elif spec.kind == "contact":
            emails, phones = _EMAIL_RE.findall(flat), _PHONE_RE.findall(flat)
            format_ok = bool(emails or phones)
            if not format_ok:
                problems.append("no email or phone number")
            digits = re.sub(r"\D", "", cited_text)
            missing = [e for e in emails if e.lower() not in cited_norm]
            missing += [p for p in phones if re.sub(r"\D", "", p)[-10:] not in digits]
            if missing:
                support = 0.0
                problems.append(f"contact details not in the cited passages: {missing}")
        threshold = SUPPORT_THRESHOLD[spec.kind]
        if support < threshold:
            problems.append(f"only {support:.0%} of the value's words are in the cited passages "
                            f"(needs {threshold:.0%})")
    support = round(support, 3)
    ok = numbers_ok and format_ok and support >= SUPPORT_THRESHOLD[spec.kind]
    return ok, support, numbers_ok, format_ok, problems


def score_confidence(citation: float, support: float, numbers_ok: bool, format_ok: bool,
                     retrieval: float, verified: bool) -> float:
    """Evidence-based confidence for a found value (formula in the module docstring)."""
    conf = 0.35 * citation + 0.35 * (support if numbers_ok else 0.0) + 0.15 * float(format_ok) + 0.15 * retrieval
    return round(conf if verified else min(conf, 0.5), 2)


def _source(r: SearchResult) -> FieldSource:
    return FieldSource(file=r.file_name, page=r.page_number, page_end=r.page_end, chunk_id=r.chunk_id,
                       citation=r.citation, doc_type=r.doc_type, addendum_number=r.addendum_number)


def better_result(new: FieldResult, old: FieldResult) -> bool:
    """A retry result replaces the earlier one only if it is better. A rejected answer that holds
    no value (unverified, value None) ranks below a clean not_found: it carries nothing to review."""
    def rank(fr: FieldResult) -> float:
        empty_reject = fr.status == "unverified" and fr.value is None
        return _STATUS_RANK["not_found"] - 0.5 if empty_reject else _STATUS_RANK[fr.status]
    a, b = rank(new), rank(old)
    return a > b or (a == b and new.confidence > old.confidence)


_better = better_result


def retry_status_wanted(status: str, retry_not_found: bool) -> bool:
    """Statuses a retry round may improve."""
    return status in ("error", "unverified") or (retry_not_found and status == "not_found")


def retry_eligible(status: str, new_chunks: set[str]) -> bool:
    """Errors are retried even on the same evidence (transient / format failures); other
    statuses only when a wider search found passages the field has not seen."""
    return status == "error" or bool(new_chunks)


def substantive_change(change_type: Optional[str], cited_text: Optional[str]) -> tuple[bool, str]:
    """Is the addendum value a real change? Needs the model's label AND change wording in the
    cited addendum passage. Returns (ok, reason)."""
    kind = (change_type or "").strip().lower()
    if kind not in SUBSTANTIVE_CHANGES:
        return False, f"not a substantive change (change={kind or 'missing'})"
    if cited_text is not None and not _CHANGE_MARKER_RE.search(cited_text):
        return False, "cited addendum passage has no change wording"
    return True, kind


def apply_reconciliation(fields: dict[str, FieldResult], specs: list[FieldSpec],
                         latest: dict[str, FieldResult], original: dict[str, FieldResult],
                         trace: ExtractionTrace, change_types: Optional[dict[str, str]] = None,
                         addendum_texts: Optional[dict[str, str]] = None) -> list[AddendumChange]:
    """Compare validated addendum values with validated original values (mutates `fields`).

    A change is recorded only when (1) the addendum value and the original value both passed
    evidence validation, (2) the change is substantive - labelled as a change and the cited
    addendum passage contains change wording (`substantive_change`), and (3) the values differ
    in substance (`substantively_same` ignores rewording and date/time formats). The field then
    takes the latest addendum value; both citations are kept. Anything less is traced and not
    recorded. Without a validated original value no change is recorded, but a validated,
    substantive addendum value may still fill an unverified / missing field.
    """
    changes: list[AddendumChange] = []
    for s in specs:
        a, b, cur = latest.get(s.name), original.get(s.name), fields[s.name]
        a_ok = a is not None and a.status == "verified" and all(x.doc_type == ADDENDUM for x in a.sources)
        b_ok = b is not None and b.status == "verified" and all(x.doc_type != ADDENDUM for x in b.sources)
        if not a_ok:
            trace.event("reconcile", field=s.name, outcome="no_validated_addendum_value",
                        addendum_status=a.status if a else None)
            continue
        cited = (" ".join(addendum_texts.get(x.chunk_id, "") for x in a.sources)
                 if addendum_texts is not None else None)
        ok, why = substantive_change((change_types or {}).get(s.name), cited)
        if not ok:
            trace.event("reconcile", field=s.name, outcome="not_substantive", reason=why, addendum_value=a.value)
            continue
        num = max((x.addendum_number for x in a.sources if x.addendum_number), default=None)
        label = f"Addendum {num}" if num else "an addendum"
        if b_ok and not substantively_same(a.value, b.value):
            changes.append(AddendumChange(
                field=s.name, value=a.value, addendum_number=num, sources=a.sources,
                previous_value=b.value, previous_sources=b.sources,
                notes=f"Changed by {label}; original value cited from {', '.join(x.citation for x in b.sources)}",
            ))
            note = f"Changed by {label} (original: {b.value!r}, {b.sources[0].citation})"
            if cur.status != "verified" or not substantively_same(cur.value, a.value):
                fields[s.name] = a.model_copy(update={"attempts": cur.attempts, "notes": note})
            elif note not in cur.notes:
                cur.notes = f"{cur.notes}; {note}" if cur.notes else note
            outcome = "changed"
        elif b_ok:
            outcome = "unchanged"  # the addendum restates the original value
        else:
            if cur.status != "verified":
                fields[s.name] = a.model_copy(update={
                    "attempts": cur.attempts, "notes": f"Value from {label}; no validated original value"})
            outcome = "addendum_only"
        trace.event("reconcile", field=s.name, outcome=outcome, addendum_value=a.value,
                    original_value=b.value if b_ok else None, original_status=b.status if b else None)
    return changes


@dataclass
class _Evidence:
    passages: list[SearchResult]
    ranks: dict[str, dict[str, int]]  # field -> {chunk_id: best rank}
    dropped: int = 0  # passages from other bids discarded

    @property
    def leak_note(self) -> str:
        return f" ({self.dropped} passage(s) from other bids discarded)" if self.dropped else ""


EvidenceSet = _Evidence  # public name for agents


@dataclass
class _CallResult:
    results: dict[str, FieldResult]
    notes: dict[str, str] = field(default_factory=dict)
    model: Optional[str] = None
    calls: int = 0  # LLM calls made (> 1 when a truncated group was split)
    changes: dict[str, str] = field(default_factory=dict)  # addendum scope: field -> "changed"/"clarification"/...

    @property
    def called(self) -> bool:
        return self.calls > 0


@dataclass
class FocusTask:
    """One focused single-field re-extraction (see `focused_retry_plan`)."""
    spec: FieldSpec
    passages: list[SearchResult]
    reason: str  # "not_found_with_cue" | "coverage_gap" | "preferred_source"
    gaps: list[str] = field(default_factory=list)

    @property
    def note(self) -> str:
        if self.reason == "coverage_gap":
            return (f"Make the value complete: a previous answer omitted {', '.join(self.gaps)}, which these "
                    "passages state. Keep everything else the passages support.")
        if self.reason == "preferred_source":
            return ("A previous answer used a seller's quote or terms of sale. Answer from these solicitation "
                    "passages only (the buyer's own instructions); return null if they do not state this field.")
        return "These passages may state this field. Answer only if they do; otherwise return null."


def _dedupe(passages: list[SearchResult]) -> list[SearchResult]:
    seen, out = set(), []
    for p in passages:
        if p.chunk_id not in seen:
            seen.add(p.chunk_id)
            out.append(p)
    return out


def _parts(value) -> list[str]:
    return [s.strip() for item in _as_items(value) for s in _SENTENCE_SPLIT_RE.split(item) if s.strip()]


def addresses_field(spec: FieldSpec, text: str) -> bool:
    """Some sentence / table cell of `text` carries the field's cue outside `cue_exclude`."""
    if not spec.cue:
        return False
    rx = re.compile(spec.cue, re.I)
    exclude = re.compile(spec.cue_exclude, re.I) if spec.cue_exclude else None
    return any(rx.search(s) and not (exclude and exclude.search(s)) for s in _SENTENCE_SPLIT_RE.split(text))


def strip_unrelated_parts(spec: FieldSpec, value) -> tuple[object, list[str]]:
    """Remove the sentences / list items of a value that match `reject_value` (e.g. subcontractor
    payment duties) -> (cleaned value or None, removed parts). The cleaned value is then checked
    against its citations like any other value."""
    if not spec.reject_value or value is None:
        return value, []
    rej = re.compile(spec.reject_value, re.I)
    if isinstance(value, list):
        kept = [v for v in value if not rej.search(v)]
        removed = [v for v in value if rej.search(v)]
        return (kept or None), removed
    parts = _parts(value)
    kept = [p for p in parts if not rej.search(p)]
    removed = [p for p in parts if rej.search(p)]
    if not removed:
        return value, []
    return (" ".join(kept).rstrip(" ;,") or None), removed


_ABSENCE_STRINGS = {"none", "no", "not applicable", "not specified", "not mentioned", "not provided",
                    "not available", "not listed", "none stated", "none found"}


def says_absent(value) -> bool:
    """'None' / 'Not specified' as the whole value: meaningful only with a citation (e.g. a
    solicitation stating no pre-bid meeting); without one it is the model finding nothing."""
    return value is not None and all(_norm(i).rstrip(".") in _ABSENCE_STRINGS for i in _as_items(value))


def drop_unsupporting_refs(spec: FieldSpec, value, refs: list[int], evidence: list[SearchResult]) -> list[int]:
    """Identifier fields: drop cited passages that contain none of the identifiers verbatim (they
    support nothing in the value). Free-text values keep every citation - their words and numbers
    may come from different passages. At least one citation always remains."""
    if spec.kind != "id" or len(refs) < 2:
        return refs
    items = [_norm(i) for i in _as_items(value)]
    return [n for n in refs if any(i in _norm(evidence[n - 1].text) for i in items)] or refs


def describes_something_else(spec: FieldSpec, value) -> bool:
    """True when every part of the value that carries the field's cue (e.g. 'invoice', 'payment
    terms') sits in a sentence matching `reject_value` (e.g. subcontractor payments) - i.e. the
    value states nothing about the field itself. A value with at least one clean cue-bearing part
    (e.g. 'Net 30 after receipt of invoice') is kept, and so is one whose rejected parts do not
    use the field's wording either ('Within 45 days of award; due date 06/10/2024': the clean
    part is the answer, the due date is stripped later)."""
    rej = re.compile(spec.reject_value, re.I)
    parts = [s for item in _as_items(value) for s in _SENTENCE_SPLIT_RE.split(item) if s.strip()]
    rejected = [s for s in parts if rej.search(s)]
    if not rejected:
        return False
    cue = re.compile(spec.cue, re.I) if spec.cue else None
    clean = [s for s in parts if not rej.search(s)]
    if not clean:
        return True
    if cue and not any(cue.search(s) for s in rejected):
        return False
    return not any(cue.search(s) if cue else True for s in clean)


def cue_passages(spec: FieldSpec, passages: list[SearchResult], ranks: dict[str, int],
                 limit: int = 3) -> list[SearchResult]:
    """Passages whose wording addresses the field (its `cue`), best-ranked first. For buyer-
    requirement fields (`prefer_doc_types`), when the solicitation's own documents address the
    field, only those are returned - a seller's quote / terms of sale may rank higher in search
    but does not state the buyer's requirement."""
    hits = [p for p in passages if addresses_field(spec, p.text)]
    hits.sort(key=lambda p: ranks.get(p.chunk_id, 10 ** 6))
    if spec.prefer_doc_types:
        preferred = [p for p in hits if p.doc_type in spec.prefer_doc_types]
        hits = preferred or hits
    return hits[:limit]


def coverage_gaps(spec: FieldSpec, value, passages: list[SearchResult],
                  fields: dict[str, FieldResult]) -> tuple[list[str], list[SearchResult]]:
    """Items the evidence states but the value omits -> (gap labels, passages stating them).

    Generic rules: the field's `coverage` regexes (e.g. a warranty stated in the evidence), and
    for Product, every verified Model_no item must be mentioned. Negation-aware: only a
    positive statement counts, on both sides - "warranty not found in provided passages" in
    the value does not cover a stated warranty, and "no warranty is offered" in a passage is
    not a warranty the value omits.
    """
    if value is None:
        return [], []
    text = " ".join(_as_items(value))
    gaps: list[str] = []
    support: list[SearchResult] = []
    for label, pattern in spec.coverage:
        rx = re.compile(pattern, re.I)
        found = [p for p in passages if states_positively(rx, p.text)]
        if found and not states_positively(rx, text):
            gaps.append(label)
            support += found
    models = fields.get("Model_no")
    if spec.name == "Product" and models is not None and models.status == "verified" and models.value:
        norm_text = normalize_text(text)
        for m in _as_items(models.value):
            if normalize_text(m) not in norm_text:
                found = [p for p in passages if normalize_text(m) in normalize_text(p.text)]
                if found:
                    gaps.append(f"model {m}")
                    support += found
    return gaps, _dedupe(support)


def focused_retry_plan(fields: dict[str, FieldResult], passages_by_field: dict[str, list[SearchResult]],
                       ranks_by_field: dict[str, dict[str, int]], limit: int = 3,
                       max_tasks: int = 8) -> list[FocusTask]:
    """Fields that deserve ONE focused single-field call over their own evidence:
      * not_found although a passage shown for the field contains its cue wording (a top
        search rank alone does not qualify - every query has a rank-1 hit);
      * verified but incomplete (`coverage_gaps`), re-asked with the cited passages plus the
        passages stating the missing items.
    Fields without such evidence are not retried."""
    tasks: list[FocusTask] = []
    for spec in FIELDS:
        fr, passages = fields.get(spec.name), passages_by_field.get(spec.name, [])
        if fr is None or not passages:
            continue
        if fr.status == "not_found":
            cue = cue_passages(spec, passages, ranks_by_field.get(spec.name, {}), limit)
            if cue:
                tasks.append(FocusTask(spec, cue, "not_found_with_cue"))
        elif fr.status == "verified":
            if spec.prefer_doc_types and fr.sources and all(s.doc_type not in spec.prefer_doc_types
                                                            for s in fr.sources):
                preferred = [p for p in cue_passages(spec, passages, ranks_by_field.get(spec.name, {}), limit)
                             if p.doc_type in spec.prefer_doc_types]
                if preferred:  # cited only seller documents, but the solicitation addresses the field
                    tasks.append(FocusTask(spec, preferred, "preferred_source"))
                    continue
            gaps, support = coverage_gaps(spec, fr.value, passages, fields)
            if gaps:
                cited_ids = {s.chunk_id for s in fr.sources}
                cited = [p for p in passages if p.chunk_id in cited_ids]
                tasks.append(FocusTask(spec, _dedupe(cited + support[:limit]), "coverage_gap", gaps))
    return tasks[:max_tasks]


def accept_focused(task: FocusTask, old: FieldResult, new: FieldResult, fields: dict[str, FieldResult]) -> bool:
    """A focused result replaces the earlier one only if it is verified (grounded in the cited
    passages); for a coverage gap it must also close part of the gap without dropping most of
    what the earlier value said."""
    if new.status != "verified":
        return False
    if task.reason == "not_found_with_cue":
        return True
    if task.reason == "preferred_source":
        return bool(new.sources) and all(s.doc_type in task.spec.prefer_doc_types for s in new.sources)
    after, _ = coverage_gaps(task.spec, new.value, task.passages, fields)
    if len(after) >= len(task.gaps):
        return False
    old_words = set(_content_words(" ".join(_as_items(old.value))))
    new_words = set(_content_words(" ".join(_as_items(new.value))))
    return not old_words or len(old_words & new_words) / len(old_words) >= 0.6


def reconciliation_candidates(latest: dict[str, FieldResult], change_types: dict[str, str],
                              addendum_texts: dict[str, str]) -> list[tuple[str, list[FieldSpec]]]:
    """Fields whose original value is worth extracting: the addendum value is validated, cited
    from addendums and a substantive change. Grouped like `reconciliation_groups`."""
    out = []
    for g, specs in reconciliation_groups():
        keep = []
        for s in specs:
            a = latest.get(s.name)
            if not (a and a.status == "verified" and a.sources and all(x.doc_type == ADDENDUM for x in a.sources)):
                continue
            cited = " ".join(addendum_texts.get(x.chunk_id, "") for x in a.sources)
            if substantive_change(change_types.get(s.name), cited)[0]:
                keep.append(s)
        if keep:
            out.append((g, keep))
    return out


def reconciliation_groups() -> list[tuple[str, list[FieldSpec]]]:
    """Addendum-sensitive fields per field group: reconciliation asks per group, so each
    request stays small and gets the evidence of its own fields."""
    out = []
    for g in GROUPS:
        specs = [s for s in fields_in(g) if s.addendum_sensitive]
        if specs:
            out.append((g, specs))
    return out


# ----------------------------------------------------------------------- service
class ExtractionService:
    """Extracts all 20 fields for a bid. `retrieval` needs `.search(query, top_k, filters, mode)`;
    `llm` needs `.complete(messages, model=None) -> LLMResponse` (inject stubs in tests)."""

    def __init__(self, retrieval, llm, *, per_query: int = 4, addendum_per_query: int = 2,
                 max_passages: int = 14, passage_chars: int = EVIDENCE_PASSAGE_CHARS, mode: Optional[str] = None,
                 model: Optional[str] = None, max_retries: int = 1, retry_not_found: bool = False,
                 reconcile_addendums: bool = True, max_split_depth: int = 2, focused_retries: bool = True,
                 max_focus_calls: int = 8, max_compact_calls: int = 3):
        self.max_split_depth = max(0, max_split_depth)
        # compact retries of truncated single-field calls, per bid run (per trace)
        self.max_compact_calls = max(0, max_compact_calls)
        self._compact_used: dict[int, int] = {}
        self.focused_retries = focused_retries
        self.max_focus_calls = max(0, max_focus_calls)
        self.retrieval = retrieval
        self.llm = llm
        self.per_query = per_query
        self.addendum_per_query = addendum_per_query
        self.max_passages = max_passages
        self.passage_chars = passage_chars
        self.mode = mode
        self.model = model
        self.max_retries = max(0, max_retries)
        self.retry_not_found = retry_not_found
        self.reconcile_addendums = reconcile_addendums
        # request fingerprint -> (outcome, result): an identical prompt over identical evidence
        # is never sent twice (only transient provider errors are not remembered)
        self._requests: dict[tuple, tuple[str, Optional[_CallResult]]] = {}
        self._requests_lock = threading.Lock()

    # -------------------------------------------------------------- evidence
    def _search(self, query: str, filters: SearchFilters, top_k: int) -> list[SearchResult]:
        try:
            return self.retrieval.search(query, top_k=top_k, filters=filters, mode=self.mode)
        except (ValueError, LookupError) as exc:  # bad mode/filter, empty index...
            log.warning("extraction search failed for %r: %s", query, exc)
            return []

    def gather_evidence(self, specs: list[FieldSpec], bid_id: str, *, scale: int = 1, extra_queries: bool = False,
                        scope: Optional[str] = None) -> _Evidence:
        """Evidence for a group of fields, restricted to `bid_id`.

        scale         multiplies the per-query result count (wider search for retries)
        extra_queries also searches with the field description
        scope         None: all documents (+ addendum searches for addendum-sensitive fields)
                      "original": non-addendum documents only | "addendum": addendums only
        """
        per_field: dict[str, list[SearchResult]] = {}
        ranks: dict[str, dict[str, int]] = {}
        dropped = 0
        for spec in specs:
            queries = list(spec.queries) + ([spec.description] if extra_queries else [])
            k = self.per_query * scale
            if scope == ADDENDUM:
                searches = [(q, SearchFilters(bid_id=bid_id, doc_type=ADDENDUM), k) for q in queries]
            elif scope == "original":
                searches = [(q, SearchFilters(bid_id=bid_id), 2 * k) for q in queries]  # addendums dropped below
            else:
                searches = [(q, SearchFilters(bid_id=bid_id), k) for q in queries]
                if spec.addendum_sensitive:
                    searches += [(q, SearchFilters(bid_id=bid_id, doc_type=ADDENDUM), self.addendum_per_query * scale)
                                 for q in queries[:1]]
            ranked: list[tuple[int, int, SearchResult]] = []  # (rank, query index, hit)
            for qi, (query, flt, top_k) in enumerate(searches):
                for r in self._search(query, flt, top_k):
                    if r.bid_id != bid_id:  # never let another bid's evidence in
                        dropped += 1
                        continue
                    if scope == "original" and r.doc_type == ADDENDUM:
                        continue
                    ranks.setdefault(spec.name, {})
                    ranks[spec.name][r.chunk_id] = min(ranks[spec.name].get(r.chunk_id, r.rank), r.rank)
                    ranked.append((r.rank, qi, r))
            # interleave the field's queries by rank: every query's best hit comes before any
            # query's second hit (concatenating per query let a later query's rank-1 hit fall
            # behind the group cap)
            per_field[spec.name] = [r for _, _, r in sorted(ranked, key=lambda x: (x[0], x[1]))]
        # round-robin over fields so every field keeps some evidence under the cap; fields that
        # hold several values (documents, products, ...) take two passages per round
        cap = self.max_passages * scale
        passages: list[SearchResult] = []
        seen: set[str] = set()
        queues = [(BY_NAME[name].multi if name in BY_NAME else False, list(h)) for name, h in per_field.items()]
        while queues and len(passages) < cap:
            for entry in list(queues):
                multi, q = entry
                for _ in range(2 if multi else 1):
                    while q and q[0].chunk_id in seen:
                        q.pop(0)
                    if not q:
                        break
                    r = q.pop(0)
                    seen.add(r.chunk_id)
                    passages.append(r.model_copy(update={"rank": len(passages) + 1}))
                    if len(passages) >= cap:
                        break
                if not q:
                    queues.remove(entry)
                if len(passages) >= cap:
                    break
        return _Evidence(passages, ranks, dropped)

    # -------------------------------------------------------------- one LLM call
    def extract_fields(self, specs: list[FieldSpec], ev: _Evidence, trace: ExtractionTrace, *, label: str,
                       attempt: int = 1, scope: Optional[str] = None, note: Optional[str] = None) -> _CallResult:
        """Ask the model for `specs` over `ev` (one LLM call) and validate every field."""
        return self._call(specs, ev, trace, label=label, attempt=attempt, scope=scope, note=note)

    def _split(self, specs: list[FieldSpec], ev: _Evidence, trace: ExtractionTrace, *, label: str, attempt: int,
               scope: Optional[str], depth: int, model: Optional[str]) -> _CallResult:
        """A group's JSON did not fit in the output budget (reasoning models spend part of it
        thinking): ask again for each half, each with only the passages retrieved for its own
        fields. Bounded by `max_split_depth` (one truncated call -> at most 2^(d+1)-1 calls)."""
        mid = (len(specs) + 1) // 2
        parts = [specs[:mid], specs[mid:]]
        trace.event("split", call=label, attempt=attempt, reason="truncated", depth=depth + 1,
                    parts=[[s.name for s in p] for p in parts])
        merged = _CallResult({}, model=model, calls=1)  # the truncated call itself
        for i, part in enumerate(parts, start=1):
            keep = set().union(*(ev.ranks.get(s.name, {}).keys() for s in part))
            sub_passages = [p for p in ev.passages if p.chunk_id in keep] or ev.passages
            sub_ev = _Evidence([p.model_copy(update={"rank": n}) for n, p in enumerate(sub_passages, start=1)],
                               {s.name: ev.ranks.get(s.name, {}) for s in part}, ev.dropped)
            sub = self._call(part, sub_ev, trace, label=f"{label}/part{i}", attempt=attempt, scope=scope,
                             depth=depth + 1)
            merged.results.update(sub.results)
            merged.notes.update(sub.notes)
            merged.changes.update(sub.changes)
            merged.calls += sub.calls
            merged.model = merged.model or sub.model
        return merged

    def _take_compact(self, trace: ExtractionTrace) -> bool:
        with self._requests_lock:
            used = self._compact_used.get(id(trace), 0)
            if used >= self.max_compact_calls:
                return False
            self._compact_used[id(trace)] = used + 1
            return True

    def _compact(self, spec: FieldSpec, ev: _Evidence, trace: ExtractionTrace, *, label: str, attempt: int,
                 scope: Optional[str], depth: int, note: Optional[str]) -> _CallResult:
        """One bounded retry for a single field whose answer hit the output limit: only the
        field's best passages (cue passages first) and a brevity note. The request differs from
        the truncated one (extra note, usually fewer passages), so it is never an identical
        repeat, and a compact call is never compacted again."""
        ranks = ev.ranks.get(spec.name, {})
        cues = cue_passages(spec, ev.passages, ranks, limit=COMPACT_PASSAGES)
        cue_ids = {p.chunk_id for p in cues}
        rest = sorted((p for p in ev.passages if p.chunk_id not in cue_ids), key=lambda p: ranks.get(p.chunk_id, 10 ** 6))
        keep = (cues + rest)[:COMPACT_PASSAGES]
        compact_note = f"{note}\n\n{COMPACT_NOTE}" if note else COMPACT_NOTE
        trace.event("compact_retry", call=label, attempt=attempt, field=spec.name, reason="truncated",
                    passages_before=len(ev.passages), passages_after=len(keep))
        sub_ev = _Evidence([p.model_copy(update={"rank": n}) for n, p in enumerate(keep, start=1)],
                           {spec.name: ranks}, ev.dropped)
        return self._call([spec], sub_ev, trace, label=f"{label}/compact", attempt=attempt, scope=scope,
                          depth=depth, note=compact_note, compact=True)

    def _call(self, specs: list[FieldSpec], ev: _Evidence, trace: ExtractionTrace, *, label: str,
              attempt: int = 1, scope: Optional[str] = None, depth: int = 0, note: Optional[str] = None,
              compact: bool = False) -> _CallResult:
        """Ask the model for `specs` over `ev` and validate every field."""
        trace.event("retrieve", call=label, attempt=attempt, scope=scope, fields=[s.name for s in specs],
                    passages=evidence_refs(ev.passages), foreign_dropped=ev.dropped)
        if not ev.passages:
            note = f"{NOT_FOUND_NOTE}: no passages retrieved for this bid{ev.leak_note}"
            trace.event("validate", call=label, attempt=attempt, outcome="no_evidence", fields=[s.name for s in specs])
            return _CallResult({s.name: FieldResult(status="not_found", notes=note, attempts=attempt) for s in specs})

        def all_error(reason: str, model=None, calls: int = 1) -> _CallResult:
            trace.event("llm_call", call=label, attempt=attempt, outcome="error", error=reason[:300])
            return _CallResult({s.name: FieldResult(status="error", notes=reason, attempts=attempt) for s in specs},
                               model=model, calls=calls)

        key = (scope, note, tuple(s.name for s in specs), tuple(p.chunk_id for p in ev.passages), self.model)
        with self._requests_lock:
            seen = self._requests.get(key)
        if seen is not None:  # identical prompt over identical evidence: do not send it again
            outcome, cached = seen
            trace.event("llm_call", call=label, attempt=attempt, outcome="not_repeated", previous_outcome=outcome)
            if outcome == "ok" and cached is not None:
                return _CallResult({k: v.model_copy(update={"attempts": attempt}) for k, v in cached.results.items()},
                                   notes=dict(cached.notes), model=cached.model, calls=0, changes=dict(cached.changes))
            return all_error(f"Identical request already made (outcome: {outcome}); not repeated", calls=0)

        def remember(outcome: str, result: Optional[_CallResult] = None) -> None:
            with self._requests_lock:
                self._requests[key] = (outcome, result)

        t0 = time.time()
        try:
            resp = self.llm.complete(build_extraction_messages(specs, ev.passages, self.passage_chars, scope, note),
                                     model=self.model)
        except LLMError as exc:
            return all_error(f"LLM call failed: {type(exc).__name__}: {exc}")
        except Exception as exc:  # never crash the whole bid on one group
            log.exception("extraction LLM call failed")
            return all_error(f"LLM call failed: {type(exc).__name__}: {exc}")
        usage = getattr(resp, "usage", None)
        call_info = dict(call=label, attempt=attempt, model=resp.model, finish_reason=resp.finish_reason,
                         latency_s=getattr(resp, "latency_s", round(time.time() - t0, 3)),
                         prompt_tokens=getattr(usage, "prompt_tokens", None),
                         completion_tokens=getattr(usage, "completion_tokens", None))
        if getattr(resp, "truncated", False):
            trace.event("llm_call", **call_info, outcome="truncated")
            remember("truncated")
            if len(specs) > 1 and depth < self.max_split_depth:
                return self._split(specs, ev, trace, label=label, attempt=attempt, scope=scope, depth=depth,
                                   model=resp.model)
            limit_note = (f"Model output stopped at the token limit (finish_reason={resp.finish_reason}, "
                          f"prompt_tokens={call_info['prompt_tokens']}, "
                          f"completion_tokens={call_info['completion_tokens']})")
            if len(specs) == 1 and not compact and self._take_compact(trace):
                sub = self._compact(specs[0], ev, trace, label=label, attempt=attempt, scope=scope, depth=depth,
                                    note=note)
                for fr in sub.results.values():
                    if fr.status == "error":
                        fr.notes = f"{limit_note}; compact retry: {fr.notes}"
                sub.calls += 1  # the truncated call itself
                sub.model = sub.model or resp.model
                return sub
            if compact:
                limit_note += "; compact retry also truncated"
            elif len(specs) == 1:
                limit_note += f"; compact retry budget ({self.max_compact_calls} per bid) used up"
            return all_error(limit_note, resp.model)
        try:
            obj = parse_json_object(resp.text)
        except ValueError as exc:
            trace.event("llm_call", **call_info, outcome="invalid_json")
            remember("invalid_json")
            return all_error(f"Model returned invalid JSON: {exc}", resp.model)
        trace.event("llm_call", **call_info, outcome="ok")

        by_key = {_key(k): v for k, v in obj.items()}
        out = _CallResult({}, model=resp.model, calls=1)
        for spec in specs:
            entry = obj.get(spec.name, by_key.get(_key(spec.name)))
            fr = self._field_result(spec, entry, ev.passages, ev.ranks.get(spec.name, {}), ev.leak_note)
            fr.attempts = attempt
            out.results[spec.name] = fr
            if isinstance(entry, dict) and entry.get("notes"):
                out.notes[spec.name] = str(entry["notes"])
            if scope == ADDENDUM and isinstance(entry, dict) and entry.get("change") is not None:
                out.changes[spec.name] = str(entry["change"]).strip().lower()
            trace.event("validate", call=label, attempt=attempt, field=spec.name, status=fr.status,
                        confidence=fr.confidence, refs=[s.chunk_id for s in fr.sources],
                        notes=fr.notes[:300], **({"change": out.changes.get(spec.name)} if scope == ADDENDUM else {}))
        remember("ok", out)
        return out

    def _field_result(self, spec: FieldSpec, entry, evidence: list[SearchResult],
                      field_ranks: dict[str, int], leak_note: str) -> FieldResult:
        if entry is None:
            return FieldResult(status="error", notes="Model output omitted this field")
        if not isinstance(entry, dict):
            return FieldResult(status="error", notes=f"Malformed field entry: {str(entry)[:80]!r}")
        model_notes = str(entry.get("notes") or "").strip()
        value = normalize_value(entry.get("value"))
        raw_refs = entry.get("refs") or []
        raw_refs = raw_refs if isinstance(raw_refs, list) else [raw_refs]
        valid, invalid = [], []
        for x in raw_refs:
            n = int(x) if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit()) else None
            (valid if n is not None and 1 <= n <= len(evidence) else invalid).append(n if n is not None else x)
        valid = list(dict.fromkeys(valid))
        if not valid and says_absent(value):  # an uncited "None" is the model finding nothing
            value = None
        if value is None:
            coverage = min(1.0, len(field_ranks) / self.per_query)
            reason = model_notes or NOT_FOUND_NOTE
            if not reason.lower().startswith("not found"):
                reason = f"{NOT_FOUND_NOTE}: {reason}"
            return FieldResult(status="not_found", confidence=round(0.6 * coverage, 2), notes=reason + leak_note)
        if spec.reject_value and describes_something_else(spec, value):
            return FieldResult(status="not_found", confidence=0.0, notes=(
                f"{NOT_FOUND_NOTE}: the model's value {value!r} describes a different clause (e.g. subcontractor "
                f"payments or the bid due date), not this field" + leak_note))
        value, removed = strip_unrelated_parts(spec, value)  # mixed value: keep only the field's own parts
        if removed:
            model_notes = (f"{model_notes}; " if model_notes else "") + \
                f"removed unrelated clause(s) from the value: {removed!r}"

        if not valid:
            return FieldResult(
                status="unverified", value=None, confidence=0.0,
                notes=(f"Rejected: the model's value {value!r} cites no valid evidence passage "
                       f"(refs {raw_refs!r})" + (f"; model note: {model_notes}" if model_notes else "")),
            )
        if removed:  # drop citations to passages that only contain the removed kind of clause
            relevant = [n for n in valid if addresses_field(spec, evidence[n - 1].text)]
            valid = relevant or valid
        valid = drop_unsupporting_refs(spec, value, valid, evidence)
        cited = [evidence[n - 1] for n in valid]
        ok, support, numbers_ok, format_ok, problems = validate_value(spec, value, valid, cited, evidence)
        if invalid:
            problems.insert(0, f"fabricated or unknown evidence refs {invalid!r}")
        verified = ok and not invalid
        best = min((field_ranks[r.chunk_id] for r in cited if r.chunk_id in field_ranks), default=None)
        retrieval = 1.0 / (1 + (best - 1) / 3) if best else 0.25
        citation = 1.0 if not invalid else 0.5
        notes = "; ".join(([model_notes] if model_notes else []) + (["Unverified: " + "; ".join(problems)] if problems else []))
        return FieldResult(
            value=value, sources=[_source(r) for r in cited], status="verified" if verified else "unverified",
            confidence=score_confidence(citation, support, numbers_ok, format_ok, retrieval, verified),
            notes=notes,
        )

    # -------------------------------------------------------------- focused retries
    def focus_call(self, task: FocusTask, ranks: dict[str, int], trace: ExtractionTrace) -> _CallResult:
        """One single-field call over the task's passages only."""
        ev = _Evidence([p.model_copy(update={"rank": n}) for n, p in enumerate(task.passages, start=1)],
                       {task.spec.name: ranks})
        trace.event("focus", field=task.spec.name, reason=task.reason, gaps=task.gaps,
                    passages=[p.chunk_id for p in task.passages])
        return self._call([task.spec], ev, trace, label=f"focus/{task.spec.name}", attempt=2, note=task.note)

    @staticmethod
    def apply_focus(task: FocusTask, res: _CallResult, fields: dict[str, FieldResult], notes: dict[str, str],
                    trace: ExtractionTrace) -> bool:
        name = task.spec.name
        old, new = fields[name], res.results.get(name)
        accepted = new is not None and accept_focused(task, old, new, fields)
        if accepted:
            new.attempts = 2
            fields[name] = new
            if name in res.notes:
                notes[name] = res.notes[name]
        else:
            what = new.status if new is not None else "no result"
            old.notes = (f"{old.notes}; " if old.notes else "") + \
                f"Focused retry ({task.reason}) did not yield a better verified value ({what}); kept"
            if task.reason == "preferred_source" and old.status == "verified":
                # the value rests only on seller documents although the solicitation addresses the
                # field: never leave it marked verified
                old.status = "unverified"
                old.confidence = min(old.confidence, 0.5)
                old.notes += "; cited only seller documents while the solicitation addresses this field"
        trace.event("focus", field=name, outcome="accepted" if accepted else "kept",
                    previous_status=old.status, new_status=new.status if new else None)
        return accepted

    def focus(self, fields: dict[str, FieldResult], notes: dict[str, str],
              passages_by_field: dict[str, list[SearchResult]], ranks_by_field: dict[str, dict[str, int]],
              trace: ExtractionTrace) -> int:
        """Run `focused_retry_plan` (at most `max_focus_calls` single-field calls)."""
        if not self.focused_retries:
            return 0
        calls = 0
        for task in focused_retry_plan(fields, passages_by_field, ranks_by_field, max_tasks=self.max_focus_calls):
            res = self.focus_call(task, ranks_by_field.get(task.spec.name, {}), trace)
            calls += res.calls
            self.apply_focus(task, res, fields, notes, trace)
        return calls

    # -------------------------------------------------------------- bid
    def extract_bid(self, bid_id: str, trace: Optional[ExtractionTrace] = None) -> BidExtraction:
        trace = trace if trace is not None else ExtractionTrace(bid_id)
        trace.event("plan", groups={g: [s.name for s in fields_in(g)] for g in GROUPS},
                    max_retries=self.max_retries, retry_not_found=self.retry_not_found,
                    reconcile_addendums=self.reconcile_addendums)
        fields: dict[str, FieldResult] = {}
        notes: dict[str, str] = {}
        shown: dict[str, set[str]] = {}  # field -> chunk ids the model has seen for it
        passages_by_field: dict[str, list[SearchResult]] = {}
        ranks_by_field: dict[str, dict[str, int]] = {}
        calls, model = 0, None

        for group in GROUPS:
            specs = fields_in(group)
            ev = self.gather_evidence(specs, bid_id)
            res = self._call(specs, ev, trace, label=group)
            calls += res.calls
            model = model or res.model
            fields.update(res.results)
            notes.update(res.notes)
            for s in specs:
                shown[s.name] = {r.chunk_id for r in ev.passages}
                passages_by_field[s.name] = ev.passages
                ranks_by_field[s.name] = ev.ranks.get(s.name, {})

        calls += self.focus(fields, notes, passages_by_field, ranks_by_field, trace)

        for attempt in range(2, self.max_retries + 2):
            failing = [s for s in FIELDS if retry_status_wanted(fields[s.name].status, self.retry_not_found)]
            if not failing:
                break
            trace.event("retry", attempt=attempt, candidates=[s.name for s in failing])
            any_call = False
            for group in GROUPS:
                specs = [s for s in failing if s.group == group]
                if not specs:
                    continue
                ev = self.gather_evidence(specs, bid_id, scale=attempt, extra_queries=True)
                eligible, skipped = [], []
                for s in specs:
                    new = set(ev.ranks.get(s.name, {})) - shown[s.name]
                    (eligible if retry_eligible(fields[s.name].status, new) else skipped).append(s)
                for s in skipped:
                    fields[s.name].notes = (fields[s.name].notes + "; " if fields[s.name].notes else "") + \
                        f"Not retried (attempt {attempt}): no additional evidence"
                if skipped:
                    trace.event("retry", attempt=attempt, group=group, skipped=[s.name for s in skipped],
                                reason="no additional evidence")
                if not eligible:
                    continue
                res = self._call(eligible, ev, trace, label=f"{group}/retry", attempt=attempt)
                calls += res.calls
                any_call = any_call or res.called
                model = model or res.model
                for s in eligible:
                    shown[s.name] |= {r.chunk_id for r in ev.passages}
                    old, new_fr = fields[s.name], res.results[s.name]
                    keep = new_fr if _better(new_fr, old) else old
                    keep.attempts = attempt
                    fields[s.name] = keep
                    if keep is new_fr and s.name in res.notes:
                        notes[s.name] = res.notes[s.name]
                    trace.event("retry", attempt=attempt, field=s.name, previous_status=old.status,
                                new_status=new_fr.status, kept=keep.status)
            if not any_call:
                break

        changes: list[AddendumChange] = []
        if self.reconcile_addendums:
            changes, used = self._reconcile(fields, bid_id, trace)
            calls += used

        fields = {name: fields[name] for name in FIELD_NAMES}  # stable assignment order
        record = BidExtraction(bid_id=bid_id, fields=fields, addendum_changes=changes,
                               validation=self._summary(fields), model=model, llm_calls=calls)
        trace.event("done", validation=record.validation.model_dump(), llm_calls=calls,
                    addendum_changes=[c.field for c in changes])
        return record

    # -------------------------------------------------------------- addendum reconciliation
    def _reconcile(self, fields: dict[str, FieldResult], bid_id: str,
                   trace: ExtractionTrace) -> tuple[list[AddendumChange], int]:
        """Per field group: latest addendum values (addendum passages only); then, only for
        fields with a validated substantive addendum value, the original values (original
        documents only); then `apply_reconciliation`."""
        latest: dict[str, FieldResult] = {}
        changes_by_field: dict[str, str] = {}
        texts: dict[str, str] = {}
        used, any_passages = 0, False
        for group, specs in reconciliation_groups():
            ev = self.gather_evidence(specs, bid_id, scope=ADDENDUM)
            if not ev.passages:
                continue
            any_passages = True
            res = self._call(specs, ev, trace, label=f"reconcile/addendum/{group}", scope=ADDENDUM)
            used += res.calls
            latest.update(res.results)
            changes_by_field.update(res.changes)
            texts.update({p.chunk_id: p.text for p in ev.passages})
        if not any_passages:
            trace.event("reconcile", outcome="skipped", reason="no addendum passages for this bid")
            return [], 0
        original: dict[str, FieldResult] = {}
        for group, specs in reconciliation_candidates(latest, changes_by_field, texts):
            ev = self.gather_evidence(specs, bid_id, scope="original")
            res = self._call(specs, ev, trace, label=f"reconcile/original/{group}", scope="original")
            used += res.calls
            original.update(res.results)
        all_specs = [s for _, specs in reconciliation_groups() for s in specs]
        return apply_reconciliation(fields, all_specs, latest, original, trace, changes_by_field, texts), used

    @staticmethod
    def summarize(fields: dict[str, FieldResult]) -> ValidationSummary:
        return ExtractionService._summary(fields)

    @staticmethod
    def _summary(fields: dict[str, FieldResult]) -> ValidationSummary:
        st = [f.status for f in fields.values()]
        return ValidationSummary(
            passed=st.count("verified"), unverified=st.count("unverified"), error=st.count("error"),
            failed=st.count("unverified") + st.count("error"), not_found=st.count("not_found"),
        )


def write_extraction(record: BidExtraction, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.bid_id}.json"
    # error notes may quote provider/exception messages: never let a key reach the file
    path.write_text(redact(record.model_dump_json(indent=2)), encoding="utf-8")
    return path


__all__ = ["ExtractionService", "FIELDS", "build_extraction_messages", "parse_json_object", "same_value",
           "score_confidence", "validate_value", "write_extraction", "BY_NAME"]
