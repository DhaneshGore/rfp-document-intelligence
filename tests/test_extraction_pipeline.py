"""Extraction pipeline - offline: bounded retries, addendum reconciliation, traces and the
`extract` CLI. LLMs are stubs; retrieval is either the real engine (fake embedder) or a
list-backed fake that returns the first top_k passages of the bid."""
import json

import pytest

import main
from extraction.service import ExtractionService
from extraction.trace import ExtractionTrace
from schemas import LLMResponse, LLMUsage, SearchResult
from tests.conftest import make_pdf
from tests.test_extraction import ANSWERS, ORIGINAL, NeedleLLM


def sr(i, text, bid="Bid", doc_type="rfp", file="RFP.pdf", page=1, add=None):
    return SearchResult(rank=i, score=1.0, chunk_id=f"{bid}:{file}:{i}", text=text, bid_id=bid, file_name=file,
                        doc_type=doc_type, page_number=page, page_end=page, addendum_number=add,
                        source_path=f"/{file}", citation=f"{file}, p. {page}")


class ListRetrieval:
    """Ignores the query: returns the first `top_k` passages of the bid matching the doc_type filter,
    so a wider (retry) search deterministically reveals more passages."""

    def __init__(self, passages):
        self.passages = passages

    def search(self, query, top_k=5, filters=None, mode=None):
        ps = [p for p in self.passages
              if p.bid_id == filters.bid_id and (not filters.doc_type or p.doc_type == filters.doc_type)]
        return [p.model_copy(update={"rank": i}) for i, p in enumerate(ps[:top_k], start=1)]


NOISE = sr(1, "General information about this bid. Read all sections carefully.")
BIDNO = sr(2, "Solicitation Number: Z-100. Issued by the Port Authority.", page=2)


class GuessingLLM(NeedleLLM):
    """Like NeedleLLM, but when the right passage is missing it still answers "Z-100" for Bid Number
    citing passage 1 (a wrong citation -> unverified)."""

    def complete(self, messages, model=None):
        resp = super().complete(messages, model)
        obj = json.loads(resp.text)
        if "Bid Number" in obj and obj["Bid Number"]["value"] is None:
            obj["Bid Number"] = {"value": "Z-100", "refs": [1], "notes": ""}
        return self._resp(json.dumps(obj))


class FlakyLLM(NeedleLLM):
    """Returns invalid JSON the first time it sees a given set of fields, then answers normally."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = set()

    def complete(self, messages, model=None):
        resp = super().complete(messages, model)
        keys = tuple(self.calls[-1]["keys"])
        if keys not in self.seen:
            self.seen.add(keys)
            return self._resp("Sorry, I cannot produce JSON right now.")
        return resp


BID_ANSWER = {"Bid Number": ("Z-100", ["Solicitation Number: Z-100"])}


# ----------------------------------------------------------------------- retries
def growing(n=20):
    """Distinct passages: a wider (retry) search always reveals new evidence."""
    return [sr(i, f"Section {i}: general bid information, part {i}.", page=i) for i in range(1, n + 1)]


def test_error_fields_are_retried_and_recover():
    llm = FlakyLLM(BID_ANSWER)
    trace = ExtractionTrace("Bid")
    # first pass sees only NOISE; the wider retry search adds BIDNO (new evidence -> new request)
    rec = ExtractionService(ListRetrieval([NOISE, BIDNO]), llm, per_query=1, max_passages=1).extract_bid("Bid", trace)
    f = rec.fields["Bid Number"]
    assert f.status == "verified" and f.attempts == 2 and f.value == "Z-100"
    assert all(x.status != "error" for x in rec.fields.values())
    outcomes = [(e["call"], e["outcome"]) for e in trace.events if e["stage"] == "llm_call"]
    assert ("identity", "invalid_json") in outcomes and ("identity/retry", "ok") in outcomes
    assert rec.llm_calls == 10  # 5 groups, each failed once and was retried once


def test_unverified_field_retried_with_additional_evidence():
    llm = GuessingLLM(BID_ANSWER)
    rec = ExtractionService(ListRetrieval([NOISE, BIDNO]), llm, per_query=1, max_passages=1).extract_bid("Bid")
    f = rec.fields["Bid Number"]
    assert f.status == "verified" and f.attempts == 2 and f.sources[0].page == 2
    first, retry = llm.calls[0], next(c for c in llm.calls if c["keys"] == ["Bid Number"])
    assert len(first["passages"]) == 1 and len(retry["passages"]) == 2  # wider search found the passage


def test_unverified_field_not_retried_without_additional_evidence():
    llm = GuessingLLM(BID_ANSWER)
    trace = ExtractionTrace("Bid")
    rec = ExtractionService(ListRetrieval([NOISE]), llm, per_query=4).extract_bid("Bid", trace)
    f = rec.fields["Bid Number"]
    assert f.status == "unverified" and f.attempts == 1 and "Not retried (attempt 2): no additional evidence" in f.notes
    assert len(llm.calls) == 5  # no retry call was made
    assert any(e["stage"] == "retry" and e.get("skipped") == ["Bid Number"] for e in trace.events)


@pytest.mark.parametrize("max_retries", [0, 1, 3])
def test_retries_are_bounded(max_retries):
    llm = NeedleLLM(text="not json at all")
    rec = ExtractionService(ListRetrieval(growing()), llm, per_query=1, max_passages=1,
                            max_retries=max_retries).extract_bid("Bid")
    assert len(llm.calls) == rec.llm_calls == 5 * (1 + max_retries)
    assert all(f.status == "error" and f.attempts == 1 + max_retries for f in rec.fields.values())


def test_identical_request_is_never_repeated():
    # the retry search finds nothing new -> the same prompt over the same evidence is not re-sent
    llm = NeedleLLM(text="not json at all")
    trace = ExtractionTrace("Bid")
    rec = ExtractionService(ListRetrieval([NOISE]), llm, max_retries=3).extract_bid("Bid", trace)
    assert len(llm.calls) == rec.llm_calls == 5
    assert all(f.status == "error" for f in rec.fields.values())
    assert any(e.get("outcome") == "not_repeated" for e in trace.events if e["stage"] == "llm_call")


def test_successful_identical_request_is_reused_not_resent():
    from extraction.fields import fields_in
    from extraction.service import EvidenceSet

    llm = NeedleLLM(BID_ANSWER)
    svc = ExtractionService(ListRetrieval([BIDNO]), llm)
    ev = svc.gather_evidence(fields_in("identity"), "Bid")
    first = svc.extract_fields(fields_in("identity"), ev, ExtractionTrace("Bid"), label="identity")
    again = svc.extract_fields(fields_in("identity"), EvidenceSet(ev.passages, ev.ranks), ExtractionTrace("Bid"),
                               label="identity", attempt=2)
    assert len(llm.calls) == 1 and first.calls == 1 and again.calls == 0
    assert again.results["Bid Number"].value == "Z-100" and again.results["Bid Number"].attempts == 2


def test_not_found_is_retried_only_when_enabled_and_new_evidence_exists():
    def run(**kw):
        llm = NeedleLLM(BID_ANSWER)
        rec = ExtractionService(ListRetrieval([NOISE, BIDNO]), llm, per_query=1, max_passages=1, **kw).extract_bid("Bid")
        return rec, llm

    rec, llm = run()
    assert rec.fields["Bid Number"].status == "not_found" and len(llm.calls) == 5
    rec, llm = run(retry_not_found=True)
    assert rec.fields["Bid Number"].status == "verified" and rec.fields["Bid Number"].attempts == 2
    # one bounded round: the wider search revealed a new passage to every group (the fake
    # retriever ignores the query), so each group was retried exactly once
    assert len(llm.calls) == 10 and "Bid Number" in llm.calls[5]["keys"]


def test_not_found_not_retried_when_wider_search_finds_nothing_new():
    llm = NeedleLLM(BID_ANSWER)
    rec = ExtractionService(ListRetrieval([NOISE]), llm, retry_not_found=True).extract_bid("Bid")
    assert len(llm.calls) == 5 and all(f.status == "not_found" for f in rec.fields.values())
    assert "no additional evidence" in rec.fields["Bid Number"].notes


# ----------------------------------------------------------------------- truncation -> split
class BudgetLLM(NeedleLLM):
    """Simulates a reasoning model with a small output budget: any request for more than
    `max_fields` fields is cut off (finish_reason=length)."""

    def __init__(self, *a, max_fields=2, **k):
        super().__init__(*a, **k)
        self.max_fields = max_fields

    def complete(self, messages, model=None):
        resp = super().complete(messages, model)
        if len(self.calls[-1]["keys"]) > self.max_fields:
            return LLMResponse(text='{"Due Date": {"value": "May', model="stub", latency_s=0.0,
                               usage=LLMUsage(completion_tokens=2048), finish_reason="length")
        return resp


def test_truncated_group_is_split_and_recovers(indexed):
    # 3 fields fit (the live model's 4-field groups used 1200-1960 of 2048 tokens); the 5-6 field
    # groups and the 10-field reconciliation calls do not
    llm = BudgetLLM(ANSWERS, max_fields=3, scoped={"original": ORIGINAL})
    trace = ExtractionTrace("BidA")
    rec = ExtractionService(indexed.retrieval, llm, per_query=10, mode="hybrid").extract_bid("BidA", trace)
    for name in ANSWERS:
        assert rec.fields[name].status == "verified", (name, rec.fields[name].notes)
    assert rec.validation.error == 0 and [c.field for c in rec.addendum_changes] == ["Due Date"]
    splits = [e for e in trace.events if e["stage"] == "split"]
    assert {e["call"] for e in splits} >= {"dates_logistics", "commercial_legal", "product", "identity",
                                           "reconcile/addendum/dates_logistics", "reconcile/addendum/product"}
    assert rec.llm_calls == len(llm.calls)  # every split call is counted
    answered = [c for c in llm.calls if len(c["keys"]) <= 3]
    assert answered and len(answered) < len(llm.calls)  # cut-off calls were retried as smaller parts


def test_split_is_bounded_and_single_fields_are_not_split():
    llm = BudgetLLM(BID_ANSWER, max_fields=0)  # every call is cut off
    rec = ExtractionService(ListRetrieval([NOISE, BIDNO]), llm, max_retries=0, reconcile_addendums=False,
                            max_split_depth=2).extract_bid("Bid")
    # groups of 6/5/4/4/1 fields: 7 + 7 + 7 + 7 + 1 calls at most (1 + 2 + 4 per group, single field 1),
    # plus the per-bid budget of compact retries for single fields that still hit the limit
    assert rec.llm_calls == len(llm.calls) == 7 + 7 + 7 + 7 + 1 + 3
    assert all(f.status == "error" and "token limit" in f.notes for f in rec.fields.values())


# ----------------------------------------------------------------------- truncated single field -> compact retry
class ThinkingLLM(NeedleLLM):
    """A reasoning model that spends the whole output budget thinking unless the prompt carries
    the brevity note (or always, with `always=True`)."""

    def __init__(self, *a, always=False, **k):
        super().__init__(*a, **k)
        self.always = always
        self.notes = []

    def complete(self, messages, model=None):
        from extraction.service import COMPACT_NOTE
        resp = super().complete(messages, model)
        compact = COMPACT_NOTE in messages[1]["content"]
        self.notes.append(compact)
        if self.always or not compact:
            return LLMResponse(text='{"Bid Number": {"val', model="stub", latency_s=0.0,
                               usage=LLMUsage(prompt_tokens=1863, completion_tokens=2048), finish_reason="length")
        return resp


def bidno_passages():
    return [BIDNO] + [sr(i, f"Section {i}: general bid information.", page=i) for i in range(3, 12)]


def single_field_run(llm, **kw):
    from extraction.fields import BY_NAME
    svc = ExtractionService(ListRetrieval(bidno_passages()), llm, per_query=12, **kw)
    trace = ExtractionTrace("Bid")
    ev = svc.gather_evidence([BY_NAME["Bid Number"]], "Bid")
    return svc.extract_fields([BY_NAME["Bid Number"]], ev, trace, label="identity/part1/part1"), trace


def test_truncated_single_field_gets_one_compact_retry_and_recovers():
    llm = ThinkingLLM(BID_ANSWER)
    res, trace = single_field_run(llm)
    f = res.results["Bid Number"]
    assert f.status == "verified" and f.value == "Z-100"
    assert res.calls == len(llm.calls) == 2 and llm.notes == [False, True]
    first, second = llm.calls
    assert len(second["passages"]) <= 4 < len(first["passages"])  # fewer passages, not an identical request
    assert any("Z-100" in t for t in second["passages"].values())  # the cue passage is kept
    (ev,) = [e for e in trace.events if e["stage"] == "compact_retry"]
    assert ev["field"] == "Bid Number" and ev["passages_after"] <= 4
    calls = [(e["call"], e["outcome"]) for e in trace.events if e["stage"] == "llm_call"]
    assert calls == [("identity/part1/part1", "truncated"), ("identity/part1/part1/compact", "ok")]


def test_compact_retry_is_bounded_and_error_is_recorded_accurately():
    llm = ThinkingLLM(BID_ANSWER, always=True)
    res, _ = single_field_run(llm)
    f = res.results["Bid Number"]
    assert res.calls == len(llm.calls) == 2  # never a third request
    assert f.status == "error"
    assert "finish_reason=length" in f.notes and "completion_tokens=2048" in f.notes and "prompt_tokens=1863" in f.notes
    assert "compact retry also truncated" in f.notes


def test_compact_retry_budget_per_bid():
    llm = ThinkingLLM(BID_ANSWER, always=True)
    res, trace = single_field_run(llm, max_compact_calls=0)
    assert len(llm.calls) == 1 and "compact retry budget (0 per bid) used up" in res.results["Bid Number"].notes
    assert not [e for e in trace.events if e["stage"] == "compact_retry"]


def test_split_halves_get_their_own_evidence():
    p_bid = sr(1, "Solicitation Number: Z-100", page=1)
    p_date = sr(2, "Proposals are due May 1, 2025 at 2:00 PM CST.", page=2)

    class FieldRetrieval(ListRetrieval):
        def search(self, query, top_k=5, filters=None, mode=None):
            pick = p_bid if "number" in query.lower() else p_date
            return [pick.model_copy(update={"rank": 1})]

    llm = BudgetLLM({"Bid Number": ("Z-100", ["Z-100"])}, max_fields=2)
    ExtractionService(FieldRetrieval([]), llm, reconcile_addendums=False, max_retries=0).extract_bid("Bid")
    part = next(c for c in llm.calls if c["keys"] == ["Bid Number", "Title"])
    assert any("Z-100" in t for t in part["passages"].values())


# ----------------------------------------------------------------------- addendum reconciliation
BASE = sr(1, "SUBMISSION\nProposals are due May 1, 2025 at 2:00 PM CST.", file="RFP-9.pdf", page=3)
ADD1 = sr(2, "ADDENDUM No. 1\nThe due date is extended to May 8, 2025 at 2:00 PM CST.", doc_type="addendum",
          file="Addendum 1.pdf", add=1)
ADD2 = sr(3, "ADDENDUM No. 2\nThe due date is extended to May 15, 2025 at 2:00 PM CST.", doc_type="addendum",
          file="Addendum 2.pdf", add=2)
LATEST = ("May 15, 2025 at 2:00 PM CST", ["extended to May 15, 2025"])
FIRST = ("May 1, 2025 at 2:00 PM CST", ["due May 1, 2025"])


def reconcile_run(main_answer, original=FIRST, latest=LATEST, passages=(BASE, ADD1, ADD2), change=None):
    llm = NeedleLLM({"Due Date": main_answer},
                    scoped={"addendum": {"Due Date": latest} if latest else {},
                            "original": {"Due Date": original} if original else {}},
                    change_types={"Due Date": change} if change is not None else None)
    trace = ExtractionTrace("Bid")
    rec = ExtractionService(ListRetrieval(list(passages)), llm).extract_bid("Bid", trace)
    return rec, llm, trace


def test_reconciliation_records_previous_and_latest_value():
    rec, _, _ = reconcile_run(LATEST)
    (c,) = rec.addendum_changes
    assert (c.field, c.value, c.previous_value, c.addendum_number) == (
        "Due Date", "May 15, 2025 at 2:00 PM CST", "May 1, 2025 at 2:00 PM CST", 2)
    assert [s.file for s in c.sources] == ["Addendum 2.pdf"] and [s.file for s in c.previous_sources] == ["RFP-9.pdf"]
    f = rec.fields["Due Date"]
    assert f.value == c.value and f.status == "verified" and "Changed by Addendum 2" in f.notes


def test_reconciliation_corrects_a_value_that_missed_the_addendum():
    rec, _, trace = reconcile_run(FIRST)  # main pass answered with the original date
    f = rec.fields["Due Date"]
    assert f.value == "May 15, 2025 at 2:00 PM CST" and f.sources[0].file == "Addendum 2.pdf"
    assert "original: 'May 1, 2025 at 2:00 PM CST'" in f.notes and len(rec.addendum_changes) == 1
    assert any(e["stage"] == "reconcile" and e.get("outcome") == "changed" for e in trace.events)


@pytest.mark.parametrize("original", [None, ("May 2, 2025 at 2:00 PM CST", ["due May 1, 2025"])])
def test_no_change_recorded_without_a_validated_original(original):
    # original not found, or an original value its cited passage does not support
    rec, _, trace = reconcile_run(LATEST, original=original)
    assert rec.addendum_changes == []
    assert rec.fields["Due Date"].value == "May 15, 2025 at 2:00 PM CST"
    assert any(e["stage"] == "reconcile" and e.get("outcome") == "addendum_only" for e in trace.events)


def test_no_change_recorded_without_a_validated_addendum_value():
    rec, _, _ = reconcile_run(FIRST, latest=("May 22, 2025 at 2:00 PM CST", ["extended to May 15, 2025"]))
    assert rec.addendum_changes == [] and rec.fields["Due Date"].value == "May 1, 2025 at 2:00 PM CST"


def _addendum(text):
    return sr(2, f"ADDENDUM No. 1\n{text}", doc_type="addendum", file="Addendum 1.pdf", add=1)


@pytest.mark.parametrize("case,passage,latest,change,outcome", [
    # restatement: no change wording in the cited addendum passage
    ("restated", "The due date remains May 1, 2025 at 2:00 PM CST.",
     ("May 1, 2025 at 2:00 PM CST", ["remains May 1, 2025"]), "changed", "not_substantive"),
    # clarification: the model labels it as such (Q&A answer about format, value unchanged)
    ("clarification", "Answer: electronic responses may be uploaded as one PDF; the due date is still "
     "May 1, 2025 at 2:00 PM CST.", ("May 1, 2025 at 2:00 PM CST", ["still May 1, 2025"]), "clarification",
     "not_substantive"),
    # rewording / reformatting: change wording present but the same date and time
    ("rewording", "The due date is changed to 05/01/2025 at 2:00 PM CST to correct a typo in the form.",
     ("05/01/2025 at 2:00 PM CST", ["changed to 05/01/2025"]), "changed", "unchanged"),
    # ambiguous: no change label at all
    ("ambiguous", "The due date is extended to May 15, 2025 at 2:00 PM CST.",
     ("May 15, 2025 at 2:00 PM CST", ["extended to May 15, 2025"]), "", "not_substantive"),
])
def test_non_substantive_addendum_values_are_not_changes(case, passage, latest, change, outcome):
    rec, _, trace = reconcile_run(FIRST, latest=latest, passages=(BASE, _addendum(passage)), change=change)
    assert rec.addendum_changes == [], case
    assert rec.fields["Due Date"].value == FIRST[0]  # the original value stands
    assert any(e["stage"] == "reconcile" and e.get("field") == "Due Date" and e.get("outcome") == outcome
               for e in trace.events), case


@pytest.mark.parametrize("passage,latest", [
    ("The due date is extended to May 15, 2025 at 2:00 PM CST.", ("May 15, 2025 at 2:00 PM CST", ["May 15, 2025"])),
    ("The bid opening is postponed; proposals are now due 05/22/2025 at 10:00 AM CST.",
     ("05/22/2025 at 10:00 AM CST", ["due 05/22/2025"])),
])
def test_genuine_change_is_recorded_with_both_citations(passage, latest):
    rec, _, _ = reconcile_run(FIRST, latest=latest, passages=(BASE, _addendum(passage)))
    (c,) = rec.addendum_changes
    assert c.value == latest[0] and c.previous_value == FIRST[0] and c.addendum_number == 1
    assert [s.file for s in c.sources] == ["Addendum 1.pdf"] and [s.file for s in c.previous_sources] == ["RFP-9.pdf"]


def test_substantively_same_handles_formats_and_rewording():
    from extraction.service import substantively_same

    assert substantively_same("27-JUN-2024 14:00:00", "June 27, 2024 at 2:00 PM")
    assert not substantively_same("27-JUN-2024 14:00:00", "July 9, 2024 at 2:00 PM CST")
    assert not substantively_same("June 27, 2024 2:00 PM", "June 27, 2024 4:00 PM")
    assert substantively_same("Electronic via the iSupplier portal or manual submission",
                              "Manual submission or electronic via the iSupplier portal")
    assert not substantively_same("5% bid bond required", "10% bid bond required")


def test_bid_without_addendums_skips_reconciliation():
    rec, llm, trace = reconcile_run(FIRST, passages=(BASE,))
    assert len(llm.calls) == 5 and rec.addendum_changes == []
    assert any(e["stage"] == "reconcile" and e.get("outcome") == "skipped" for e in trace.events)


# ----------------------------------------------------------------------- traces
@pytest.fixture
def indexed(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    return engine


def test_trace_records_every_stage_with_evidence_refs_but_no_text(indexed):
    trace = ExtractionTrace("BidA")
    ExtractionService(indexed.retrieval, NeedleLLM(ANSWERS, scoped={"original": ORIGINAL}), per_query=10,
                      mode="hybrid").extract_bid("BidA", trace)
    stages = trace.stages()
    assert stages[0] == "plan" and stages[-1] == "done"
    assert {"retrieve", "llm_call", "validate", "reconcile"} <= set(stages)
    retrieve = next(e for e in trace.events if e["stage"] == "retrieve")
    assert retrieve["passages"] and set(retrieve["passages"][0]) == {"n", "chunk_id", "file", "page", "doc_type",
                                                                   "addendum_number"}
    call = next(e for e in trace.events if e["stage"] == "llm_call")
    assert {"model", "latency_s", "prompt_tokens", "completion_tokens", "finish_reason", "outcome"} <= set(call)
    lines = [json.loads(x) for x in trace.to_jsonl().splitlines()]
    assert [e["seq"] for e in lines] == list(range(1, len(lines) + 1))
    assert "I hereby affirm" not in trace.to_jsonl()  # passage text is never written


def test_trace_and_json_redact_the_api_key(engine, monkeypatch, tmp_path):
    import config
    from extraction.service import write_extraction

    secret = "sk-or-v1-" + "b2" * 32
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    config.get_settings.cache_clear()
    try:
        trace = ExtractionTrace("Bid")
        rec = ExtractionService(ListRetrieval([NOISE]), NeedleLLM(exc=RuntimeError(f"upstream echoed {secret}")),
                                max_retries=0).extract_bid("Bid", trace)
        assert secret not in trace.to_jsonl() and "[REDACTED]" in trace.to_jsonl()
        text = write_extraction(rec, tmp_path).read_text(encoding="utf-8")
        assert secret not in text and "[REDACTED]" in text
    finally:
        config.get_settings.cache_clear()


# ----------------------------------------------------------------------- CLI
@pytest.fixture
def cli_engine(engine, monkeypatch):
    import services.container

    monkeypatch.setattr(services.container, "get_engine", lambda: engine)
    return engine


def test_cli_extract_writes_json_and_trace(cli_engine, bid_folder, tmp_path, capsys):
    cli_engine.indexing.index_folder(bid_folder)
    cli_engine.llm = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    out = tmp_path / "out"
    assert main.main(["extract", "--bid", "BidA", "--out", str(out)]) == 0
    data = json.loads((out / "BidA.json").read_text(encoding="utf-8"))
    assert data["bid_id"] == "BidA" and len(data["fields"]) == 20
    assert data["fields"]["Due Date"]["status"] == "verified" and data["addendum_changes"][0]["previous_value"]
    events = [json.loads(x) for x in (out / "BidA.trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["stage"] == "plan" and events[-1]["stage"] == "done"
    printed = capsys.readouterr().out
    assert "JSON:" in printed and "verified" in printed and "llm_calls=9" in printed


def test_cli_extract_new_folder_path_indexes_then_extracts(cli_engine, settings, tmp_path):
    folder = settings.bids_root / "Lakeside_Schools_2026"
    folder.mkdir()
    make_pdf(folder / "IFB 26-7 Tablets.pdf", ["INVITATION FOR BID IFB 26-7\nLakeside School District"])
    cli_engine.llm = NeedleLLM({"Bid Number": ("IFB 26-7", ["INVITATION FOR BID IFB 26-7"])})
    out = tmp_path / "out"
    assert main.main(["extract", "--bid", str(folder), "--out", str(out)]) == 0
    data = json.loads((out / "Lakeside_Schools_2026.json").read_text(encoding="utf-8"))
    bn = data["fields"]["Bid Number"]
    assert bn["status"] == "verified" and bn["sources"][0]["file"] == "IFB 26-7 Tablets.pdf"


def test_cli_unknown_bid_is_rejected_before_any_llm_call(cli_engine, bid_folder, tmp_path, capsys):
    cli_engine.indexing.index_folder(bid_folder)
    cli_engine.llm = llm = NeedleLLM(ANSWERS)
    out = tmp_path / "out"
    assert main.main(["extract", "--bid", "NoSuchBid", "--out", str(out)]) == 2
    assert llm.calls == [] and not out.exists()
    assert "Indexed bids: BidA" in capsys.readouterr().err


def test_cli_without_api_key_does_not_call_the_real_llm(cli_engine, bid_folder, tmp_path, capsys):
    from services.llm_service import OpenRouterLLM

    cli_engine.indexing.index_folder(bid_folder)
    assert isinstance(cli_engine.llm, OpenRouterLLM) and not cli_engine.settings.openrouter_api_key
    out = tmp_path / "out"
    assert main.main(["extract", "--bid", "BidA", "--out", str(out)]) == 2
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err and not out.exists()


def test_cli_exit_code_1_when_fields_errored(cli_engine, bid_folder, tmp_path):
    cli_engine.indexing.index_folder(bid_folder)
    cli_engine.llm = NeedleLLM(text="<html>502 Bad Gateway</html>")
    assert main.main(["extract", "--bid", "BidA", "--out", str(tmp_path), "--max-retries", "0"]) == 1
    data = json.loads((tmp_path / "BidA.json").read_text(encoding="utf-8"))
    assert data["validation"]["error"] == 20
