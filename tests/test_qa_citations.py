"""Citation generation and grounded-answer behaviour (LLM mocked)."""
import pytest

from schemas import LLMResponse, LLMUsage, SearchFilters
from services.llm_service import LLMRateLimitError
from services.qa_service import NOT_FOUND_MESSAGE, QAService, build_messages, resolve_citations


class StubLLM:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.messages = text, exc, None

    def complete(self, messages, model=None):
        self.messages = messages
        if self.exc:
            raise self.exc
        return LLMResponse(text=self.text, model=model or "stub/model:free", latency_s=0.01,
                           usage=LLMUsage(total_tokens=42))


@pytest.fixture
def indexed(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    return engine


def test_resolve_citations_maps_refs_to_real_sources(indexed):
    ev = indexed.retrieval.search("bid bond due date memory affidavit", top_k=3, mode="bm25")
    assert len(ev) == 3
    cits, invalid = resolve_citations("Bond is 5% [1]. Also see [1][2] and [9].", ev)
    assert [c.ref for c in cits] == [1, 2]
    assert invalid == [9]
    assert cits[0].file_name == ev[0].file_name and cits[0].page_number == ev[0].page_number
    assert cits[0].chunk_id == ev[0].chunk_id and cits[0].text == ev[0].text


def test_prompt_contains_numbered_evidence_with_sources(indexed):
    ev = indexed.retrieval.search("due date", top_k=2, mode="bm25")
    msgs = build_messages("When is it due?", ev)
    assert msgs[0]["role"] == "system" and "NOT_FOUND" in msgs[0]["content"]
    assert f"[1] (bid: {ev[0].bid_id}; file: {ev[0].file_name}" in msgs[1]["content"]
    assert "type: addendum #1" in msgs[1]["content"] or "type: rfp" in msgs[1]["content"]


def test_ask_returns_cited_answer(indexed):
    stub = StubLLM("A 5% bid bond is required [1].")
    qa = QAService(indexed.retrieval, stub)
    ans = qa.ask("Is a bid bond required?", filters=SearchFilters(bid_id="BidA"), mode="bm25")
    assert ans.found and ans.citations and ans.error is None
    c = ans.citations[0]
    assert c.file_name == "RFP-555 Widget Supply.pdf"
    assert c.page_number <= 2 <= c.page_end  # bond clause is on page 2 of the fixture
    assert ans.usage.total_tokens == 42


def test_ask_not_found_from_llm(indexed):
    qa = QAService(indexed.retrieval, StubLLM("NOT_FOUND: The information was not found in the indexed documents."))
    ans = qa.ask("What is the insurance limit?", mode="bm25")
    assert not ans.found and ans.answer == NOT_FOUND_MESSAGE and ans.citations == []


def test_ask_without_evidence_skips_llm(engine):
    stub = StubLLM("should not be called")
    ans = QAService(engine.retrieval, stub).ask("anything at all", mode="hybrid")
    assert not ans.found and ans.answer == NOT_FOUND_MESSAGE
    assert stub.messages is None


def test_ask_flags_uncited_answer(indexed):
    ans = QAService(indexed.retrieval, StubLLM("It is due in April.")).ask("due date?", mode="bm25")
    assert ans.found and not ans.citations and "no valid citations" in ans.error


def test_ask_llm_error_is_reported_not_raised(indexed):
    qa = QAService(indexed.retrieval, StubLLM(exc=LLMRateLimitError("rate limited")))
    ans = qa.ask("due date?", mode="bm25")
    assert not ans.found and "LLMRateLimitError" in ans.error and ans.evidence


def test_evidence_union_and_addendum_context(indexed):
    qa = QAService(indexed.retrieval, StubLLM("x"))
    ev = qa.gather_evidence("What is the revised due date?", mode="bm25")
    ids = [r.chunk_id for r in ev]
    assert len(ids) == len(set(ids))  # de-duplicated
    assert [r.rank for r in ev] == list(range(1, len(ev) + 1))
    assert any(r.doc_type == "addendum" for r in ev)  # change intent -> addendum context
    assert len(ev) <= indexed.settings.ask_max_context
    # an explicit doc_type filter is respected (no addendum injection)
    from schemas import SearchFilters as F
    only_specs = qa.gather_evidence("What is the revised memory?", filters=F(doc_type="specs"), mode="bm25")
    assert {r.doc_type for r in only_specs} <= {"specs"}


def test_ask_rejects_empty_question(indexed):
    with pytest.raises(ValueError):
        QAService(indexed.retrieval, StubLLM("x")).ask("  ")
