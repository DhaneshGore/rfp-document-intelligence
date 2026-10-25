"""Multi-agent workflow - offline integration tests (stub LLMs, fake embedder, no quota)."""
import json
import threading
import time

import pytest

import main
from agents import BidNotIndexedError, Orchestrator
from agents.contracts import ExtractionInput, RetrievalOutput, ValidationInput
from agents.state import WorkflowState
from agents.validator import ValidatorAgent
from extraction.fields import FIELD_NAMES, GROUPS
from extraction.schemas import FieldResult, FieldSource
from extraction.service import ExtractionService
from extraction.trace import ExtractionTrace
from schemas import SearchFilters
from tests.test_extraction import ANSWERS, ORIGINAL, NeedleLLM
from tests.test_extraction_pipeline import (ADD1, ADD2, BASE, BID_ANSWER, BIDNO, FIRST, LATEST, NOISE, GuessingLLM,
                                            ListRetrieval)
from tests.test_qa_citations import StubLLM


@pytest.fixture
def indexed(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    return engine


def orch(engine, llm, **kw):
    kw.setdefault("per_query", 10)
    kw.setdefault("mode", "hybrid")
    return Orchestrator(engine.retrieval, llm, qa=engine.qa, indexing=engine.indexing, **kw)


def handoffs(trace):
    return [e for e in trace.events if e["stage"] == "handoff"]


# ----------------------------------------------------------------------- successful run
def test_extraction_handoffs_state_and_parity_with_service(indexed):
    llm = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    trace = ExtractionTrace("BidA")
    state = orch(indexed, llm).run_extraction(bid_id="BidA", trace=trace)
    rec = state.final
    # same result and the same number of LLM calls as the plain service path (no extra "agent" calls)
    llm2 = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    ref = ExtractionService(indexed.retrieval, llm2, per_query=10, mode="hybrid").extract_bid("BidA")
    assert {n: (f.status, f.value) for n, f in rec.fields.items()} == {n: (f.status, f.value) for n, f in ref.fields.items()}
    assert state.llm_calls == rec.llm_calls == len(llm.calls) == ref.llm_calls == len(GROUPS) + 4
    assert [c.field for c in rec.addendum_changes] == ["Due Date"] and rec.addendum_changes[0].previous_value

    # shared state
    assert set(state.drafts) == set(FIELD_NAMES) and set(state.evidence) == set(GROUPS)
    assert all(state.shown[n] for n in ANSWERS) and state.errors == []
    assert all(v.decision in ("accept", "not_found") for v in state.verdicts.values())
    assert state.report_markdown.startswith("# Bid report: BidA")
    steps = {s.id: s.status for s in state.plan}
    assert all(steps[f"retrieve:{g}"] == steps[f"extract:{g}"] == "done" for g in GROUPS)
    assert steps["validate:1"] == steps["validate:final"] == steps["report"] == steps["reconcile:compare"] == "done"

    # every handoff is traced, followed by the agent's result
    targets = [h["target"] for h in handoffs(trace)]
    assert {"retrieval", "extraction", "validator", "reconciliation", "qa_report"} <= set(targets)
    agent_events = [e for e in trace.events if e["stage"] == "agent"]
    assert len(agent_events) == len(handoffs(trace)) and all(e["status"] == "done" for e in agent_events)
    plan = trace.events[0]
    assert plan["stage"] == "plan" and {a["name"]: a["uses_llm"] for a in plan["agents"]} == {
        "ingestion": False, "retrieval": False, "extraction": True, "validator": False,
        "reconciliation": False, "qa_report": True}
    assert "I hereby affirm" not in trace.to_jsonl()  # evidence is referenced, never copied


def test_folder_input_runs_ingestion_agent_first(engine, settings):
    from tests.conftest import make_pdf

    folder = settings.bids_root / "Riverside-2026"
    folder.mkdir()
    make_pdf(folder / "RFP RV-31.pdf", ["REQUEST FOR PROPOSAL RV-31\nRiverside County"])
    trace = ExtractionTrace("x")
    state = orch(engine, NeedleLLM({"Bid Number": ("RV-31", ["PROPOSAL RV-31"])})).run_extraction(
        folder=str(folder), trace=trace)
    assert state.bid_id == "Riverside-2026" and handoffs(trace)[0]["target"] == "ingestion"
    assert state.final.fields["Bid Number"].status == "verified"


def test_unknown_bid_stops_before_any_agent_work(indexed):
    llm = NeedleLLM(ANSWERS)
    with pytest.raises(BidNotIndexedError, match="Indexed bids: BidA"):
        orch(indexed, llm).run_extraction(bid_id="Nope")
    assert llm.calls == []


# ----------------------------------------------------------------------- parallelism
class SlowLLM(NeedleLLM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def complete(self, messages, model=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.15)
        try:
            return super().complete(messages, model)
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.parametrize("workers,parallel", [(5, True), (1, False)])
def test_independent_groups_run_in_parallel_only_when_allowed(workers, parallel):
    llm = SlowLLM(BID_ANSWER)
    state = Orchestrator(ListRetrieval([NOISE, BIDNO]), llm, workers=workers, reconcile=False,
                         focused_retries=False).run_extraction("Bid")
    assert (llm.peak > 1) is parallel
    assert state.final.fields["Bid Number"].status == "verified" and state.llm_calls == len(GROUPS)


# ----------------------------------------------------------------------- critic and retries
def test_validator_rejection_drives_a_bounded_retry():
    llm = GuessingLLM(BID_ANSWER)
    trace = ExtractionTrace("Bid")
    state = Orchestrator(ListRetrieval([NOISE, BIDNO]), llm, per_query=1, max_passages=1,
                         reconcile=False).run_extraction("Bid", trace=trace)
    f = state.final.fields["Bid Number"]
    assert f.status == "verified" and f.attempts == 2
    ids = [s.id for s in state.plan]
    assert {"retrieve:identity/retry2", "extract:identity/retry2", "validate:2"} <= set(ids)
    assert state.retry_rounds == 1 and state.verdicts["Bid Number"].decision == "accept"
    first_validation = next(e for e in trace.events if e["stage"] == "agent" and e["step"] == "validate:1")
    assert "Bid Number" in first_validation["output"]["retry"]


def test_retries_stop_at_the_limit_with_give_up():
    from tests.test_extraction_pipeline import growing

    llm = NeedleLLM(text="not json")
    state = Orchestrator(ListRetrieval(growing()), llm, per_query=1, max_passages=1, max_retries=2,
                         reconcile=False).run_extraction("Bid")
    assert state.llm_calls == len(llm.calls) == len(GROUPS) * 3 and state.retry_rounds == 2
    assert all(v.decision == "give_up" for v in state.verdicts.values())
    assert state.final.validation.error == 20


def test_critic_checks_sources_formats_and_decisions():
    src = FieldSource(file="a.pdf", page=1, chunk_id="c1", citation="a.pdf, p. 1")
    fields = {
        "Bid Number": FieldResult(value="X-1", sources=[src], status="verified", confidence=0.9),
        "Title": FieldResult(value="T", sources=[src.model_copy(update={"chunk_id": "elsewhere"})],
                             status="verified", confidence=0.9),
        "Due Date": FieldResult(value="March 3, 2025", sources=[src], status="verified", confidence=0.9),
        "Payment Terms": FieldResult(value="Net 30", status="not_found"),
        "Term of Bid": FieldResult(status="error", attempts=2),
    }
    shown = {n: {"c1"} for n in fields}
    out = ValidatorAgent().run(ValidationInput(bid_id="B", fields=fields, shown=shown, max_attempts=2),
                               WorkflowState(mode="extract"), ExtractionTrace("B"))
    assert out.verdicts["Bid Number"].decision == "accept"
    t = out.fields["Title"]
    assert t.status == "unverified" and t.confidence <= 0.5 and "not among the evidence shown" in t.notes
    assert out.downgraded == ["Title"] and out.verdicts["Title"].decision == "retry"
    assert "Due Date has no time of day" in out.warnings
    assert out.fields["Payment Terms"].value is None and out.verdicts["Payment Terms"].decision == "not_found"
    assert out.verdicts["Term of Bid"].decision == "give_up"  # attempts exhausted
    assert fields["Title"].status == "verified"  # input not mutated


# ----------------------------------------------------------------------- reconciliation
def test_reconciliation_agent_records_validated_change():
    llm = NeedleLLM({"Due Date": FIRST}, scoped={"addendum": {"Due Date": LATEST}, "original": {"Due Date": FIRST}})
    trace = ExtractionTrace("Bid")
    state = Orchestrator(ListRetrieval([BASE, ADD1, ADD2]), llm).run_extraction("Bid", trace=trace)
    (c,) = state.addendum_changes
    assert (c.value, c.previous_value, c.addendum_number) == (LATEST[0], FIRST[0], 2)
    assert state.final.fields["Due Date"].value == LATEST[0]
    steps = {s.id: s.status for s in state.plan}
    assert all(steps[k] == "done" for k in ("reconcile:retrieve_addendum:dates_logistics",
                                            "reconcile:extract_addendum:dates_logistics",
                                            "reconcile:retrieve_original:dates_logistics",
                                            "reconcile:extract_original:dates_logistics", "reconcile:compare"))
    # originals are requested only for the field with a substantive addendum change
    assert [c["keys"] for c in llm.calls if c["scope"] == "original"] == [["Due Date"]]
    assert {c["scope"] for c in llm.calls} == {None, "addendum", "original"}
    # the critic accepts reconciled values: their sources were shown to the model
    assert state.verdicts["Due Date"].decision == "accept"


# ----------------------------------------------------------------------- errors and isolation
def test_failed_agents_are_contained(indexed):
    class BrokenRetrieval:
        def __init__(self, inner):
            self.inner = inner

        def search(self, query, top_k=5, filters=None, mode=None):
            if "payment" in query:
                raise RuntimeError("index unavailable")
            return self.inner.search(query, top_k=top_k, filters=filters, mode=mode)

        def bids(self):
            return self.inner.bids()

    llm = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    trace = ExtractionTrace("BidA")
    state = Orchestrator(BrokenRetrieval(indexed.retrieval), llm, per_query=10, mode="hybrid").run_extraction(
        "BidA", trace=trace)
    assert any(e.startswith("retrieve:commercial_legal") and "index unavailable" in e for e in state.errors)
    assert state.step("extract:commercial_legal").status == "skipped"
    bad = state.final.fields["Payment Terms"]
    assert bad.status == "error" and "Retrieval agent failed" in bad.notes
    assert "Retry 2 skipped: retrieval failed" in bad.notes  # the bounded retry did not turn it into "not found"
    assert state.final.fields["Bid Number"].status == "verified"  # other groups unaffected
    assert state.final is not None and state.step("report").status == "done"


def test_extraction_agent_refuses_evidence_from_another_bid(indexed):
    o = orch(indexed, NeedleLLM(ANSWERS))
    st = WorkflowState(mode="extract", bid_id="BidA")
    with pytest.raises(ValueError, match="handed to extraction"):
        o.extractor.run(ExtractionInput(bid_id="BidA", label="identity", fields=["Bid Number"],
                                        evidence=RetrievalOutput(label="identity", bid_id="OtherBid")),
                        st, ExtractionTrace("BidA"))


# ----------------------------------------------------------------------- question mode
def test_question_mode_handoffs_match_ask(indexed):
    q = "Is a bid bond required?"
    flt = SearchFilters(bid_id="BidA")
    trace = ExtractionTrace("BidA")
    indexed.qa.llm = StubLLM("A 5% bid bond is required [1].")  # the Q&A agent answers via the engine's QAService
    state = orch(indexed, NeedleLLM()).run_qa(q, filters=flt, mode="bm25", trace=trace)
    direct = indexed.qa.__class__(indexed.retrieval, StubLLM("A 5% bid bond is required [1].")).ask(q, flt, mode="bm25")
    a = state.answer
    assert a.status == direct.status == "verified" and [c.chunk_id for c in a.citations] == [c.chunk_id for c in direct.citations]
    assert [h["target"] for h in handoffs(trace)] == ["retrieval", "qa_report"] and state.llm_calls == 1


def test_cli_extract_writes_report_and_cli_ask_uses_agents(indexed, monkeypatch, tmp_path, capsys):
    import services.container

    indexed.llm = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    monkeypatch.setattr(services.container, "get_engine", lambda: indexed)
    out = tmp_path / "out"
    assert main.main(["extract", "--bid", "BidA", "--out", str(out), "--workers", "2"]) == 0
    report = (out / "BidA.report.md").read_text(encoding="utf-8")
    assert "| Due Date | April 15, 2024 at 2:00 PM | verified |" in report and "## Addendum changes" in report
    events = [json.loads(x) for x in (out / "BidA.trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["stage"] == "handoff" and e["target"] == "reconciliation" for e in events)

    indexed.llm = StubLLM("A 5% bid bond is required [1].")
    indexed.qa.llm = indexed.llm
    tpath = tmp_path / "ask.jsonl"
    assert main.main(["ask", "Is a bid bond required?", "--bid-id", "BidA", "--mode", "bm25",
                      "--trace", str(tpath)]) == 0
    assert "status: verified" in capsys.readouterr().out
    assert [json.loads(x)["target"] for x in tpath.read_text(encoding="utf-8").splitlines()
            if json.loads(x)["stage"] == "handoff"] == ["retrieval", "qa_report"]
