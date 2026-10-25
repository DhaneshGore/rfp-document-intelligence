"""Validator / critic agent (deterministic, no LLM).

The Extraction agent already checks each value against its cited passages. The critic
reviews the draft record as a whole and decides what happens next:
  * a value must have sources, and every source must be a passage the model was actually
    shown for that field (guards against fabricated or foreign citations) - else the field
    is downgraded to unverified;
  * format consistency: null values for not_found/error, confidence in [0, 1]; a Due Date
    without a time is flagged (the assignment asks for date, time and time zone);
  * verdict per field: accept (verified), retry (error / unverified while attempts remain;
    not_found only if enabled), not_found, or give_up (attempts exhausted).
"""
from __future__ import annotations

import re

from agents.base import Agent
from agents.contracts import FieldVerdict, ValidationInput, ValidationOutput
from agents.state import WorkflowState
from extraction.service import ExtractionService, retry_status_wanted
from extraction.trace import ExtractionTrace

_TIME_RE = re.compile(r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)|\b\d{1,2}:\d{2}\b", re.I)


class ValidatorAgent(Agent):
    name = "validator"
    uses_llm = False
    input_type, output_type = ValidationInput, ValidationOutput

    def run(self, inp: ValidationInput, state: WorkflowState, trace: ExtractionTrace) -> ValidationOutput:
        fields = {k: v.model_copy(deep=True) for k, v in inp.fields.items()}
        verdicts: dict[str, FieldVerdict] = {}
        downgraded, warnings, retry = [], [], []
        for name, f in fields.items():
            reasons: list[str] = []
            if f.status in ("verified", "unverified") and f.value is not None:
                if not f.sources:
                    reasons.append("value has no sources")
                foreign = [s.chunk_id for s in f.sources if s.chunk_id not in inp.shown.get(name, set())]
                if foreign:
                    reasons.append(f"sources not among the evidence shown for this field: {foreign}")
                if reasons and f.status == "verified":
                    f.status = "unverified"
                    f.confidence = min(f.confidence, 0.5)
                    f.notes = (f"{f.notes}; " if f.notes else "") + "Critic: " + "; ".join(reasons)
                    downgraded.append(name)
            if f.status in ("not_found", "error") and f.value is not None:
                f.value = None
                reasons.append("value cleared: status has no supported value")
            f.confidence = min(max(f.confidence, 0.0), 1.0)
            if name == "Due Date" and f.status == "verified" and not _TIME_RE.search(str(f.value)):
                warnings.append("Due Date has no time of day")

            attempts_left = (not inp.final) and f.attempts < inp.max_attempts
            if f.status == "verified":
                decision = "accept"
            elif retry_status_wanted(f.status, inp.retry_not_found) and attempts_left:
                decision = "retry"
                retry.append(name)
            elif f.status == "not_found":
                decision = "not_found"
            else:
                decision = "give_up"
            verdicts[name] = FieldVerdict(field=name, decision=decision, reasons=reasons)
        return ValidationOutput(fields=fields, verdicts=verdicts, retry=retry, warnings=warnings,
                                downgraded=downgraded, summary_counts=ExtractionService.summarize(fields))
