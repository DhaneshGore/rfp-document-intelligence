"""Q&A / report agent.

* QAInput  -> QAOutput: cited answer from the Retrieval agent's evidence via
  `QAService.answer` (one LLM call; citation + grounding validation and answer status as
  in the Ask API).
* ReportInput -> ReportOutput: the final `BidExtraction` JSON record and a human-readable
  Markdown report rendered from validated fields (no LLM call; the Bid Summary field is
  produced by the Extraction agent from evidence).
"""
from __future__ import annotations

from agents.base import Agent
from agents.contracts import QAInput, QAOutput, ReportInput, ReportOutput
from agents.state import WorkflowState
from extraction.fields import FIELD_NAMES
from extraction.schemas import BidExtraction
from extraction.trace import ExtractionTrace


def _cell(value) -> str:
    if value is None:
        return "–"
    text = "; ".join(value) if isinstance(value, list) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")[:160]


def render_report(record: BidExtraction) -> str:
    v = record.validation
    lines = [
        f"# Bid report: {record.bid_id}",
        "",
        f"Validation: {v.passed} verified, {v.unverified} unverified, {v.error} errors, {v.not_found} not found. "
        f"Model: {record.model or '-'}; LLM calls: {record.llm_calls}; generated {record.generated_at}.",
        "",
    ]
    summary = record.fields.get("Bid Summary")
    if summary and summary.value and summary.status == "verified":
        lines += ["## Summary", "", _cell(summary.value), ""]
    lines += ["## Fields", "", "| Field | Value | Status | Confidence | Sources |", "|---|---|---|---|---|"]
    for name in FIELD_NAMES:
        f = record.fields[name]
        src = "; ".join(s.citation for s in f.sources) or ("–" if f.status != "not_found" else f.notes[:60])
        lines.append(f"| {name} | {_cell(f.value)} | {f.status} | {f.confidence:.2f} | {_cell(src)} |")
    if record.addendum_changes:
        lines += ["", "## Addendum changes", ""]
        for c in record.addendum_changes:
            lines.append(f"* **{c.field}**: {_cell(c.previous_value)} → {_cell(c.value)} "
                         f"(Addendum {c.addendum_number}; {', '.join(s.citation for s in c.sources)})")
    problems = [(n, f) for n, f in record.fields.items() if f.status in ("unverified", "error")]
    if problems:
        lines += ["", "## Needs review", ""]
        lines += [f"* **{n}** ({f.status}): {f.notes[:200]}" for n, f in problems]
    return "\n".join(lines) + "\n"


class QAReportAgent(Agent):
    name = "qa_report"
    uses_llm = True  # for answers; reports are rendered without the model
    input_type, output_type = (QAInput, ReportInput), (QAOutput, ReportOutput)

    def __init__(self, qa=None):
        self.qa = qa

    def run(self, inp, state: WorkflowState, trace: ExtractionTrace):
        if isinstance(inp, QAInput):
            if self.qa is None:
                raise RuntimeError("answering needs a QAService")
            return QAOutput(answer=self.qa.answer(inp.question, inp.evidence.passages, inp.model))
        if isinstance(inp, ReportInput):
            record = BidExtraction(bid_id=inp.bid_id, fields={n: inp.fields[n] for n in FIELD_NAMES},
                                   addendum_changes=inp.addendum_changes, validation=inp.validation,
                                   model=inp.model, llm_calls=inp.llm_calls)
            return ReportOutput(record=record, markdown=render_report(record))
        raise TypeError(f"unsupported input {type(inp).__name__}")
