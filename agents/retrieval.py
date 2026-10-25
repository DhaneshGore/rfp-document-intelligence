"""Retrieval agent: evidence with source metadata, always restricted to one bid (no LLM).

Field mode wraps `ExtractionService.gather_evidence` (per-field queries, addendum searches,
scopes, wider retry searches); question mode wraps `QAService.gather_evidence` (query
expansion, hybrid + rerank, addendum context). Passages from another bid are dropped.
"""
from __future__ import annotations

from agents.base import Agent
from agents.contracts import RetrievalInput, RetrievalOutput
from agents.state import WorkflowState
from extraction.fields import BY_NAME
from extraction.trace import ExtractionTrace


class RetrievalAgent(Agent):
    name = "retrieval"
    uses_llm = False
    input_type, output_type = RetrievalInput, RetrievalOutput

    def __init__(self, extraction_service, qa=None):
        self.extraction = extraction_service
        self.qa = qa

    def run(self, inp: RetrievalInput, state: WorkflowState, trace: ExtractionTrace) -> RetrievalOutput:
        if inp.question is not None:
            if self.qa is None:
                raise RuntimeError("question retrieval needs a QAService")
            passages = self.qa.gather_evidence(inp.question, inp.filters, inp.top_k, inp.mode)
            bid = inp.filters.bid_id if inp.filters else None
            kept = [p for p in passages if bid is None or p.bid_id == bid]
            return RetrievalOutput(label=inp.label, bid_id=bid, passages=kept,
                                   dropped_foreign=len(passages) - len(kept))
        if not inp.bid_id:
            raise ValueError("field retrieval needs a bid_id")
        specs = [BY_NAME[f] for f in inp.fields]
        ev = self.extraction.gather_evidence(specs, inp.bid_id, scale=inp.scale, extra_queries=inp.extra_queries,
                                             scope=inp.scope)
        return RetrievalOutput(label=inp.label, bid_id=inp.bid_id, passages=ev.passages, ranks=ev.ranks,
                               dropped_foreign=ev.dropped)
