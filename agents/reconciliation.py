"""Addendum reconciliation agent (no LLM of its own).

It states which scoped extractions it needs and the Orchestrator runs them on the Retrieval
and Extraction agents:
  1. `requests()` - latest addendum values, one request per field group (small outputs,
     focused evidence);
  2. `original_requests()` - original-document values, only for fields whose addendum value
     is validated and a substantive change (no calls for clarifications);
  3. `run()` - records a change only when both values are validated, the change is
     substantive (model label + change wording in the cited addendum passage) and the values
     differ in substance (`apply_reconciliation`).
"""
from __future__ import annotations

from agents.base import Agent
from agents.contracts import ReconciliationInput, ReconciliationOutput, ReconciliationRequest
from agents.state import WorkflowState
from extraction.schemas import FieldResult
from extraction.service import apply_reconciliation, reconciliation_candidates, reconciliation_groups
from extraction.trace import ExtractionTrace


class ReconciliationAgent(Agent):
    name = "reconciliation"
    uses_llm = False
    input_type, output_type = ReconciliationInput, ReconciliationOutput

    def requests(self) -> list[ReconciliationRequest]:
        return [ReconciliationRequest(scope="addendum", group=g, fields=[s.name for s in specs])
                for g, specs in reconciliation_groups()]

    def original_requests(self, latest: dict[str, FieldResult], change_types: dict[str, str],
                          addendum_texts: dict[str, str]) -> list[ReconciliationRequest]:
        return [ReconciliationRequest(scope="original", group=g, fields=[s.name for s in specs])
                for g, specs in reconciliation_candidates(latest, change_types, addendum_texts)]

    def run(self, inp: ReconciliationInput, state: WorkflowState, trace: ExtractionTrace) -> ReconciliationOutput:
        fields = {k: v.model_copy(deep=True) for k, v in inp.fields.items()}
        specs = [s for _, group in reconciliation_groups() for s in group]
        changes = apply_reconciliation(fields, specs, inp.latest, inp.original, trace, inp.change_types,
                                       inp.addendum_texts)
        return ReconciliationOutput(fields=fields, changes=changes)
