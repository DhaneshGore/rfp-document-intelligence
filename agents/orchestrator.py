"""Orchestrator: plans the run, owns the WorkflowState, hands typed messages to agents,
runs independent work in parallel, applies the critic's retry decisions (bounded) and
records every handoff in the trace.

Extraction mode (one bid):
    [ingest] -> check_bid -> retrieve:<group> x5 -> extract:<group> x5 (parallel)
      -> validate:1 -> (retry round k: retrieve wider -> extract (parallel) -> validate:k)*
      -> reconcile: retrieve addendum/original -> extract both scopes (parallel) -> compare
      -> validate:final -> report
Question mode:
    retrieve:question -> answer

LLM calls are made only by the Extraction agent (one per field group / retry group /
reconciliation scope) and the Q&A agent (one per question) - the same number as the plain
`ExtractionService.extract_bid` path; no call exists just to make a function an "agent".
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from agents.base import Agent
from agents.contracts import (ExtractionInput, ExtractionOutput, IngestionInput, QAInput, ReconciliationInput,
                              ReportInput, RetrievalInput, RetrievalOutput, ValidationInput)
from agents.extraction import ExtractionAgent
from agents.ingestion import IngestionAgent
from agents.qa_report import QAReportAgent
from agents.reconciliation import ReconciliationAgent
from agents.retrieval import RetrievalAgent
from agents.state import WorkflowState
from agents.validator import ValidatorAgent
from extraction.fields import BY_NAME, FIELD_NAMES, GROUPS, fields_in
from extraction.schemas import FieldResult
from extraction.service import ExtractionService, better_result, focused_retry_plan, retry_eligible
from extraction.trace import ExtractionTrace
from schemas import Answer, SearchFilters
from services.llm_service import redact


class BidNotIndexedError(LookupError):
    pass


class Orchestrator:
    def __init__(self, retrieval, llm, *, qa=None, indexing=None, max_retries: int = 1,
                 retry_not_found: bool = False, reconcile: bool = True, workers: int = 3,
                 model: Optional[str] = None, **extraction_kwargs: Any):
        self.retrieval_service = retrieval
        self.max_retries = max(0, max_retries)
        self.retry_not_found = retry_not_found
        self.reconcile = reconcile
        self.workers = max(1, workers)
        self.model = model
        self.extraction_service = ExtractionService(retrieval, llm, model=model, max_retries=self.max_retries,
                                                    retry_not_found=retry_not_found,
                                                    reconcile_addendums=reconcile, **extraction_kwargs)
        self.ingestion = IngestionAgent(indexing) if indexing is not None else None
        self.retriever = RetrievalAgent(self.extraction_service, qa)
        self.extractor = ExtractionAgent(self.extraction_service)
        self.validator = ValidatorAgent()
        self.reconciler = ReconciliationAgent()
        self.reporter = QAReportAgent(qa)

    @classmethod
    def from_engine(cls, eng, **kw: Any) -> "Orchestrator":
        return cls(eng.retrieval, eng.llm, qa=eng.qa, indexing=eng.indexing, **kw)

    def agents(self) -> list[Agent]:
        return [a for a in (self.ingestion, self.retriever, self.extractor, self.validator, self.reconciler,
                            self.reporter) if a is not None]

    # -------------------------------------------------------------- handoffs
    def _invoke(self, state: WorkflowState, trace: ExtractionTrace, step_id: str, agent: Agent, inp: Any):
        """Hand `inp` to `agent` for plan step `step_id`; failures are recorded, not raised."""
        step = state.step(step_id)
        step.status = "running"
        trace.event("handoff", step=step_id, source="orchestrator", target=agent.name, uses_llm=agent.uses_llm,
                    input=inp.summary())
        t0 = time.time()
        try:
            out = agent.run(inp, state, trace)
        except Exception as exc:  # an agent failure must not crash the run
            step.status, step.error = "failed", redact(f"{type(exc).__name__}: {exc}")[:300]
            step.latency_s = round(time.time() - t0, 3)
            state.errors.append(f"{step_id}: {step.error}")
            trace.event("agent", step=step_id, agent=agent.name, status="failed", error=step.error,
                        latency_s=step.latency_s)
            return None
        step.status, step.latency_s = "done", round(time.time() - t0, 3)
        trace.event("agent", step=step_id, agent=agent.name, status="done", latency_s=step.latency_s,
                    output=out.summary())
        return out

    def _parallel(self, state, trace, jobs: list[tuple[str, Agent, Any]]) -> list:
        """Run independent handoffs concurrently (bounded by `workers`); results in job order."""
        if len(jobs) <= 1 or self.workers == 1:
            return [self._invoke(state, trace, *j) for j in jobs]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(jobs)), thread_name_prefix="agent") as pool:
            futures = [pool.submit(self._invoke, state, trace, *j) for j in jobs]
            return [f.result() for f in futures]

    def _extract_many(self, state, trace, requests: list[tuple[str, ExtractionInput]]) -> dict[str, ExtractionOutput]:
        """Parallel Extraction-agent calls; a failed agent call turns its fields into errors."""
        outs = self._parallel(state, trace, [(sid, self.extractor, inp) for sid, inp in requests])
        result: dict[str, ExtractionOutput] = {}
        for (sid, inp), out in zip(requests, outs):
            if out is None:
                err = state.step(sid).error or "extraction agent failed"
                out = ExtractionOutput(label=inp.label, results={
                    f: FieldResult(status="error", notes=f"Extraction agent failed: {err}", attempts=inp.attempt)
                    for f in inp.fields})
            state.llm_calls += out.llm_calls
            state.model = state.model or out.model
            result[sid] = out
        return result

    # -------------------------------------------------------------- extraction mode
    def run_extraction(self, bid_id: Optional[str] = None, folder: Optional[str] = None,
                       trace: Optional[ExtractionTrace] = None) -> WorkflowState:
        if not bid_id and not folder:
            raise ValueError("run_extraction needs a bid_id or a folder")
        state = WorkflowState(mode="extract", bid_id=bid_id, folder=str(folder) if folder else None)
        trace = trace if trace is not None else ExtractionTrace(bid_id or Path(folder).name)
        if folder:
            state.add_step("ingest", "ingestion", "parse, clean, chunk and index the bid folder")
        state.add_step("check_bid", "orchestrator", "bid has indexed documents", ["ingest"] if folder else [])
        for g in GROUPS:
            state.add_step(f"retrieve:{g}", "retrieval", f"evidence for {GROUPS[g]}", ["check_bid"])
            state.add_step(f"extract:{g}", "extraction", f"extract {GROUPS[g]}", [f"retrieve:{g}"])
        state.add_step("validate:1", "validator", "critic + retry plan", [f"extract:{g}" for g in GROUPS])
        trace.event("plan", run_id=state.run_id, mode="extract", steps=state.plan_summary(),
                    agents=[{"name": a.name, "uses_llm": a.uses_llm} for a in self.agents()],
                    max_retries=self.max_retries, retry_not_found=self.retry_not_found,
                    reconcile=self.reconcile, workers=self.workers)

        # 1. ingestion (optional) and bid check
        if folder:
            if self.ingestion is None:
                raise RuntimeError("folder input needs an indexing service")
            ing = self._invoke(state, trace, "ingest", self.ingestion, IngestionInput(folder=str(folder), bid_id=bid_id))
            if ing is None:
                self._finish_skipped(state)
                raise BidNotIndexedError(f"ingestion failed: {state.step('ingest').error}")
            state.bid_id = trace.bid_id = bid_id = ing.bid_id
        self._check_bid(state, trace, bid_id)

        # 2. retrieval per group (cheap, sequential), then extraction per group (parallel)
        evidence: dict[str, RetrievalOutput] = {}
        for g in GROUPS:
            names = [s.name for s in fields_in(g)]
            out = self._invoke(state, trace, f"retrieve:{g}", self.retriever,
                               RetrievalInput(bid_id=bid_id, label=g, fields=names))
            if out is None:  # retrieval failed: an error, not "not found"; no LLM call for this group
                err = state.step(f"retrieve:{g}").error
                state.step(f"extract:{g}").status = "skipped"
                for n in names:
                    state.drafts[n] = FieldResult(status="error", notes=f"Retrieval agent failed: {err}")
                    state.shown[n] = set()
                continue
            evidence[g] = state.evidence[g] = out
            for n in names:
                state.shown[n] = {p.chunk_id for p in out.passages}
        outs = self._extract_many(state, trace, [
            (f"extract:{g}", ExtractionInput(bid_id=bid_id, label=g, fields=[s.name for s in fields_in(g)],
                                             evidence=evidence[g])) for g in GROUPS if g in evidence])
        for out in outs.values():
            state.drafts.update(out.results)
            state.notes.update(out.notes)

        # 2b. focused single-field retries: not_found despite a passage that addresses the field,
        #     or verified but incomplete (coverage gap) - one call each, own evidence only
        self._focus(state, trace, bid_id, evidence)

        # 3. critic + bounded retry rounds
        verdict = self._validate(state, trace, "validate:1")
        for attempt in range(2, self.max_retries + 2):
            retry = verdict.retry if verdict else []
            if not retry:
                break
            state.retry_rounds += 1
            trace.event("retry", attempt=attempt, candidates=retry)
            requests: list[tuple[str, ExtractionInput]] = []
            for g in GROUPS:
                names = [n for n in retry if BY_NAME[n].group == g]
                if not names:
                    continue
                rid = state.add_step(f"retrieve:{g}/retry{attempt}", "retrieval", "wider search",
                                     [f"validate:{attempt - 1}"]).id
                ev = self._invoke(state, trace, rid, self.retriever,
                                  RetrievalInput(bid_id=bid_id, label=f"{g}/retry", fields=names, scale=attempt,
                                                 extra_queries=True))
                if ev is None:  # retrieval failed again: keep the earlier results (no LLM call)
                    for n in names:
                        d = state.drafts[n]
                        d.notes = (d.notes + "; " if d.notes else "") + f"Retry {attempt} skipped: retrieval failed"
                    continue
                eligible, skipped = [], []
                for n in names:
                    new = set(ev.ranks.get(n, {})) - state.shown.get(n, set())
                    (eligible if retry_eligible(state.drafts[n].status, new) else skipped).append(n)
                for n in skipped:
                    d = state.drafts[n]
                    d.notes = (d.notes + "; " if d.notes else "") + f"Not retried (attempt {attempt}): no additional evidence"
                if skipped:
                    trace.event("retry", attempt=attempt, group=g, skipped=skipped, reason="no additional evidence")
                if eligible:
                    sid = state.add_step(f"extract:{g}/retry{attempt}", "extraction", "re-extract", [rid]).id
                    requests.append((sid, ExtractionInput(bid_id=bid_id, label=f"{g}/retry", fields=eligible,
                                                          evidence=ev, attempt=attempt)))
                    for n in eligible:
                        state.shown[n] = state.shown.get(n, set()) | {p.chunk_id for p in ev.passages}
            if not requests:
                break
            for sid, out in self._extract_many(state, trace, requests).items():
                for n, new_fr in out.results.items():
                    old = state.drafts[n]
                    keep = new_fr if better_result(new_fr, old) else old
                    keep.attempts = attempt
                    state.drafts[n] = keep
                    if keep is new_fr and n in out.notes:
                        state.notes[n] = out.notes[n]
                    trace.event("retry", attempt=attempt, field=n, previous_status=old.status,
                                new_status=new_fr.status, kept=keep.status)
            state.add_step(f"validate:{attempt}", "validator", "critic after retry", [s for s, _ in requests])
            verdict = self._validate(state, trace, f"validate:{attempt}")

        # 4. addendum reconciliation
        if self.reconcile:
            self._reconcile(state, trace, bid_id)

        # 5. final critic pass and report
        state.add_step("validate:final", "validator", "final critic pass (no retries)")
        final = self._validate(state, trace, "validate:final", final=True)
        summary = final.summary_counts if final else ExtractionService.summarize(state.drafts)
        state.add_step("report", "qa_report", "final JSON record + Markdown report", ["validate:final"])
        rep = self._invoke(state, trace, "report", self.reporter, ReportInput(
            bid_id=bid_id, fields={n: state.drafts[n] for n in FIELD_NAMES}, addendum_changes=state.addendum_changes,
            validation=summary, model=state.model, llm_calls=state.llm_calls))
        if rep is not None:
            state.final, state.report_markdown = rep.record, rep.markdown
        self._finish_skipped(state)
        trace.event("done", run_id=state.run_id, validation=summary.model_dump(), llm_calls=state.llm_calls,
                    retry_rounds=state.retry_rounds, addendum_changes=[c.field for c in state.addendum_changes],
                    errors=state.errors, steps=state.plan_summary())
        return state

    def _focus(self, state: WorkflowState, trace: ExtractionTrace, bid_id: str,
               evidence: dict[str, RetrievalOutput]) -> None:
        svc = self.extraction_service
        if not svc.focused_retries:
            return
        passages = {s.name: evidence[g].passages for g in evidence for s in fields_in(g)}
        ranks = {s.name: evidence[g].ranks.get(s.name, {}) for g in evidence for s in fields_in(g)}
        tasks = focused_retry_plan(state.drafts, passages, ranks, max_tasks=svc.max_focus_calls)
        requests = []
        for t in tasks:
            name = t.spec.name
            trace.event("focus", field=name, reason=t.reason, gaps=t.gaps, passages=[p.chunk_id for p in t.passages])
            sid = state.add_step(f"focus:{name}", "extraction", t.reason, [f"extract:{t.spec.group}"]).id
            ev = RetrievalOutput(label=f"focus/{name}", bid_id=bid_id, ranks={name: ranks.get(name, {})},
                                 passages=[p.model_copy(update={"rank": n}) for n, p in enumerate(t.passages, 1)])
            requests.append((sid, ExtractionInput(bid_id=bid_id, label=f"focus/{name}", fields=[name], evidence=ev,
                                                  attempt=2, note=t.note)))
        outs = self._extract_many(state, trace, requests)
        for t, (sid, _) in zip(tasks, requests):
            svc.apply_focus(t, outs[sid], state.drafts, state.notes, trace)

    def _check_bid(self, state: WorkflowState, trace: ExtractionTrace, bid_id: str) -> None:
        step = state.step("check_bid")
        bids = getattr(self.retrieval_service, "bids", None)
        if bids is None:  # a retrieval backend without a bid listing (e.g. test doubles)
            step.status = "skipped"
            return
        known = {b["bid_id"] for b in bids() if b.get("documents")}
        if bid_id not in known:
            step.status, step.error = "failed", f"bid '{bid_id}' has no indexed documents"
            self._finish_skipped(state)
            trace.event("agent", step="check_bid", agent="orchestrator", status="failed", error=step.error,
                        indexed_bids=sorted(known))
            raise BidNotIndexedError(f"{step.error}. Indexed bids: {', '.join(sorted(known)) or 'none'}")
        step.status = "done"

    def _validate(self, state, trace, step_id: str, final: bool = False):
        if step_id not in {s.id for s in state.plan}:
            state.add_step(step_id, "validator")
        out = self._invoke(state, trace, step_id, self.validator, ValidationInput(
            bid_id=state.bid_id, fields=state.drafts, shown=state.shown, max_attempts=1 + self.max_retries,
            retry_not_found=self.retry_not_found, final=final))
        if out is not None:
            state.drafts = out.fields
            state.verdicts = out.verdicts
        return out

    def _reconcile(self, state: WorkflowState, trace: ExtractionTrace, bid_id: str) -> None:
        """1. addendum values per field group (retrieve sequentially, extract in parallel);
        2. original values only for fields with a validated substantive addendum value;
        3. compare. Evidence for each request is restricted to its own fields and scope."""

        def scoped(requests, scope: str) -> tuple[dict[str, FieldResult], dict[str, str], dict[str, str], set[str]]:
            jobs, seen = [], set()
            for r in requests:
                rid = state.add_step(f"reconcile:retrieve_{scope}:{r.group}", "retrieval", f"{scope} passages",
                                     ["validate:1"]).id
                ev = self._invoke(state, trace, rid, self.retriever, RetrievalInput(
                    bid_id=bid_id, label=f"reconcile/{scope}/{r.group}", fields=r.fields, scope=scope))
                if ev is None or not ev.passages:
                    continue
                seen |= {p.chunk_id for p in ev.passages}
                eid = state.add_step(f"reconcile:extract_{scope}:{r.group}", "extraction", f"{scope} values",
                                     [rid]).id
                jobs.append((eid, ExtractionInput(bid_id=bid_id, label=f"reconcile/{scope}/{r.group}",
                                                  fields=r.fields, evidence=ev, scope=scope), ev))
            results, changes, texts = {}, {}, {}
            outs = self._extract_many(state, trace, [(eid, inp) for eid, inp, _ in jobs])
            for eid, inp, ev in jobs:
                results.update(outs[eid].results)
                changes.update(outs[eid].change_types)
                texts.update({p.chunk_id: p.text for p in ev.passages})
            return results, changes, texts, seen

        latest, change_types, texts, seen = scoped(self.reconciler.requests(), "addendum")
        if not seen:
            trace.event("reconcile", outcome="skipped", reason="no addendum passages for this bid")
            return
        candidates = self.reconciler.original_requests(latest, change_types, texts)
        if not candidates:
            trace.event("reconcile", outcome="no_substantive_addendum_values",
                        change_types=change_types)
        original, _, _, seen_orig = scoped(candidates, "original")
        cmp_id = state.add_step("reconcile:compare", "reconciliation", "record substantive, validated changes").id
        rec = self._invoke(state, trace, cmp_id, self.reconciler, ReconciliationInput(
            bid_id=bid_id, fields=state.drafts, latest=latest, original=original, change_types=change_types,
            addendum_texts=texts))
        if rec is not None:
            state.drafts = rec.fields
            state.addendum_changes = rec.changes
            for r in self.reconciler.requests():  # reconciled values cite these passages
                for n in r.fields:
                    state.shown[n] = state.shown.get(n, set()) | seen | seen_orig

    @staticmethod
    def _finish_skipped(state: WorkflowState) -> None:
        for s in state.plan:
            if s.status == "pending":
                s.status = "skipped"

    # -------------------------------------------------------------- question mode
    def run_qa(self, question: str, filters: Optional[SearchFilters] = None, mode: Optional[str] = None,
               top_k: Optional[int] = None, model: Optional[str] = None,
               trace: Optional[ExtractionTrace] = None) -> WorkflowState:
        bid = filters.bid_id if filters else None
        state = WorkflowState(mode="qa", bid_id=bid, question=question)
        trace = trace if trace is not None else ExtractionTrace(bid or "all-bids")
        state.add_step("retrieve:question", "retrieval", "hybrid search + rerank + addendum context")
        state.add_step("answer", "qa_report", "cited answer", ["retrieve:question"])
        trace.event("plan", run_id=state.run_id, mode="qa", steps=state.plan_summary())
        ev = self._invoke(state, trace, "retrieve:question", self.retriever, RetrievalInput(
            label="question", question=question, filters=filters, mode=mode, top_k=top_k))
        if ev is None:
            state.answer = Answer(question=question, answer="", found=False, status="error",
                                  error=f"Retrieval failed: {state.step('retrieve:question').error}")
        else:
            out = self._invoke(state, trace, "answer", self.reporter, QAInput(question=question, evidence=ev,
                                                                             model=model or self.model))
            state.answer = out.answer if out is not None else Answer(
                question=question, answer="", found=False, status="error",
                error=f"Q&A agent failed: {state.step('answer').error}")
            state.llm_calls += int(bool(state.answer.model))
        self._finish_skipped(state)
        trace.event("done", run_id=state.run_id, status=state.answer.status, llm_calls=state.llm_calls,
                    errors=state.errors, steps=state.plan_summary())
        return state
