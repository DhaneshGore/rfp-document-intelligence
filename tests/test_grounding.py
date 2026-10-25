"""Regression tests for citation parsing, grounding checks and secret redaction."""
import pytest

from schemas import LLMResponse, LLMUsage, SearchResult
from services.llm_service import LLMProviderError, redact
from services.qa_service import QAService, _ref_numbers, check_grounding, resolve_citations
from tests.test_qa_citations import StubLLM


def _r(i, text, page=1, file="a.pdf", bid="Bid1"):
    return SearchResult(rank=i, score=1.0, chunk_id=f"c{i}", text=text, bid_id=bid, file_name=file,
                        doc_type="rfp", page_number=page, page_end=page, source_path=f"/{file}",
                        citation=f"{file}, p. {page}")


@pytest.mark.parametrize("text,expected", [
    ("Due [1].", [1]),
    ("Due [1][3] and [1].", [1, 3]),
    ("Due [1, 3].", [1, 3]),
    ("Due [2-4].", [2, 3, 4]),
    ("Due [2–3].", [2, 3]),
    ("No refs here (1).", []),
])
def test_reference_formats(text, expected):
    assert _ref_numbers(text) == expected


def test_resolve_citations_with_list_format():
    ev = [_r(1, "x"), _r(2, "y", page=5, file="b.pdf"), _r(3, "z")]
    cits, invalid = resolve_citations("Fact [2, 7].", ev)
    assert [c.ref for c in cits] == [2] and cits[0].page_number == 5 and cits[0].file_name == "b.pdf"
    assert invalid == [7]


def test_grounding_accepts_values_present_in_cited_passage():
    ev = [_r(1, "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.")]
    g = check_grounding("The deadline is July 9, 2024 at 2:00 PM CST [1].", ev, ev)
    assert g == {"not_in_cited": [], "not_in_evidence": []}


def test_grounding_matches_numeric_date_formats():
    ev = [_r(1, "Closing Date: 07/09/2024 03:00 PM EDT")]
    assert check_grounding("Closes 7/9/2024 at 3:00 PM EDT [1].", ev, ev)["not_in_cited"] == []


def test_grounding_detects_wrong_citation_and_hallucination():
    a = _r(1, "Solicitation Due 27-JUN-2024 14:00:00")
    b = _r(2, "The new due date is July 9, 2024 at 2:00 PM CST.")
    g = check_grounding("Originally due June 27, 2024 [2]; now July 9, 2024 [2]. Bond is 5% [2].", [b], [a, b])
    assert "27" in g["not_in_cited"]  # in evidence, but not in the cited passage
    assert "5" in g["not_in_evidence"] and "27" not in g["not_in_evidence"]


def test_grounding_ignores_list_markers_and_citation_numbers():
    ev = [_r(1, "Mercury affidavit required.")]
    g = check_grounding("1. A Mercury affidavit is required [1].\n2. Nothing else [1].", ev, ev)
    assert g["not_in_cited"] == []


class _Retrieval:
    def __init__(self, ev):
        self.ev = ev

    def search(self, *a, **k):
        return self.ev


def test_ask_flags_ungrounded_values_and_strips_think():
    ev = [_r(1, "Delivery within 45 days of Award.", page=2, file="PORFP.pdf", bid="Bid2")]
    llm = StubLLM("<think>reasoning about 99 things</think>Delivery is required within 60 days of award [1].")
    ans = QAService(_Retrieval(ev), llm).ask("When must laptops be delivered?")
    assert "<think>" not in ans.answer and "99" not in ans.answer
    assert ans.ungrounded_values == ["60"]
    assert "60" in ans.error and ans.citations[0].page_number == 2


def test_ask_grounded_answer_has_no_error():
    ev = [_r(1, "Delivery within 45 days of Award.", page=2, file="PORFP.pdf", bid="Bid2")]
    ans = QAService(_Retrieval(ev), StubLLM("Delivery is required within 45 days of award [1].")).ask("q?")
    assert ans.found and ans.error is None and ans.ungrounded_values == []


def test_not_found_with_markdown_prefix():
    ev = [_r(1, "unrelated")]
    ans = QAService(_Retrieval(ev), StubLLM("**NOT_FOUND**: The information was not found.")).ask("q?")
    assert not ans.found and ans.citations == []


def test_html_citation_has_no_page():
    ev = [SearchResult(rank=1, score=1, chunk_id="h1", text="Closing Date: 06/10/2024", bid_id="Bid2",
                       file_name="page.html", doc_type="bid_page", page_number=None, source_path="/p.html",
                       citation="page.html")]
    ans = QAService(_Retrieval(ev), StubLLM("Bids close 06/10/2024 [1].")).ask("q?")
    assert ans.citations[0].page_number is None and ans.citations[0].citation == "page.html"


def test_redaction(settings, monkeypatch):
    import config

    secret = "sk-or-v1-" + "a1" * 32
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    config.get_settings.cache_clear()
    assert secret not in redact(f"bad key {secret} rejected")
    assert "sk-or-v1-zzzzzzzzzzzz" not in redact("other key sk-or-v1-zzzzzzzzzzzz")
    err = LLMProviderError(f"provider echoed Authorization: Bearer {secret}")
    assert secret not in str(err) and "[REDACTED]" in str(err)
    config.get_settings.cache_clear()
