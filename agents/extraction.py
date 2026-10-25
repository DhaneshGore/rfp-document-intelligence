"""Extraction agent: fills a group of fields from given evidence - one LLM call
(wraps `ExtractionService.extract_fields`: JSON prompt, ref resolution, evidence checks)."""
from __future__ import annotations

from agents.base import Agent
from agents.contracts import ExtractionInput, ExtractionOutput
from agents.state import WorkflowState
from extraction.fields import BY_NAME
from extraction.service import EvidenceSet
from extraction.trace import ExtractionTrace


class ExtractionAgent(Agent):
    name = "extraction"
    uses_llm = True
    input_type, output_type = ExtractionInput, ExtractionOutput

    def __init__(self, extraction_service):
        self.extraction = extraction_service

    def run(self, inp: ExtractionInput, state: WorkflowState, trace: ExtractionTrace) -> ExtractionOutput:
        if inp.evidence.bid_id not in (None, inp.bid_id):
            raise ValueError(f"evidence for bid {inp.evidence.bid_id!r} handed to extraction for {inp.bid_id!r}")
        ev = EvidenceSet(inp.evidence.passages, inp.evidence.ranks, inp.evidence.dropped_foreign)
        res = self.extraction.extract_fields([BY_NAME[f] for f in inp.fields], ev, trace, label=inp.label,
                                             attempt=inp.attempt, scope=inp.scope, note=inp.note)
        return ExtractionOutput(label=inp.label, results=res.results, notes=res.notes, model=res.model,
                                llm_called=res.called, llm_calls=res.calls, change_types=res.changes)
