"""Table questions end to end: caption-aware chunking, retrieval in every mode, grounded
cited answers, numeric grounding, citation validation and truncated LLM responses.

The fixture reproduces the layout problems of a real sample-tables PDF (captions parsed
as separate paragraphs, a caption-less duplicate of a table on the next page, two
captions merged into one paragraph, a look-alike table elsewhere). LLM calls are stubbed.
"""
import re

import httpx
import pytest

from ingestion.chunker import Chunker
from schemas import (Block, BlockType, DocType, DocumentMetadata, LLMResponse, LLMUsage, ParsedDocument,
                     SearchFilters, SearchResult)
from services.qa_service import QAService, check_grounding, grounding_details, _numbers
from tests.test_llm import FREE_A, make_llm

Q = "In Table 5, what were the Property, Investment, and Intangibles values for 2010, 2009, and 2008?"
BID = "TableBid"
FILE = "sample-tables.pdf"
EXPECTED = {
    "Property": {"2010": "345", "2009": "445", "2008": "222"},
    "Investment": {"2010": "567", "2009": "654", "2008": "423"},
    "Intangibles": {"2010": "423", "2009": "123", "2008": "453"},
}
ANSWER = ("In Table 5, Property was 345 (2010), 445 (2009) and 222 (2008); Investment was 567 (2010), "
          "654 (2009) and 423 (2008); Intangibles were 423 (2010), 123 (2009) and 453 (2008) [{n}].")


def text(t, page):
    return Block(type=BlockType.TEXT, text=t, page=page)


def table(header, rows, page):
    return Block(type=BlockType.TABLE, text="\n".join([header] + rows), page=page,
                 table_header=header, table_rows=rows)


T5_ROWS = ["Non-current assets", "Property | 345 | 445 | 222", "Investment | 567 | 654 | 423",
           "Intangibles | 423 | 123 | 453", "Current assets", "Trade and other receivables | 435 | 634 | 231",
           "Cash and cash equivalents | 524 | 123 | 482", "Other | 223 | 211 | 254"]


def sample_blocks():
    return [
        Block(type=BlockType.HEADING, text="Design and build accessible PDF tables", page=1, level=1),
        text("Table 4: table 3 with column headers added", 2),
        table("Role | Actor", ["Daniel Radcliffe | Main character", "Rupert Grint | Sidekick 1"], 2),
        text("Table 5: year-end financial statement (£, thousands)", 2),
        table("2010 | 2009 | 2008", T5_ROWS, 2),
        text("Table 6: a table with a more serious headings problem "
             "Table 7: year-end statement, non-current assets (£, thousands)", 2),
        table("Rainfall (inches) | Americas | Asia", ["2010", "Average | 104 | 201", "24 hour high | 15 | 26"], 2),
        # Table 7's rows continue on page 3 without a caption: same values as Table 5
        table("Non-current assets | 2010 | 2009 | 2008", T5_ROWS[1:4], 3),
        text("Table 8: year-end statement, current assets (£, thousands)", 3),
        table("Current assets | 2010 | 2009 | 2008", ["Cash | 524 | 123 | 482"], 3),
        text("The figures below are unaudited and provided for information only.", 4),
        table("Item | Amount", ["Grants | 12", "Fees | 30"], 4),
        text("Table 24: year-end financial statement (£, thousands)", 9),
        table("2010 | 2009 | 2008", ["Buildings | 345 | 445 | 222", "Investment | 567 | 654 | 423",
                                    "Intangibles | 423 | 123 | 453"], 9),
    ]


def sample_doc(path="/x/" + FILE, blocks=None):
    md = DocumentMetadata(bid_id=BID, file_name=FILE, source_path=path, doc_type=DocType.OTHER,
                          content_hash="h" * 64, num_pages=9, parser="docling")
    return ParsedDocument(metadata=md, blocks=blocks or sample_blocks())


@pytest.fixture
def tables_engine(engine, tmp_path, monkeypatch):
    """Index the sample layout through the real chunker, vector store, BM25 and manifest."""
    f = tmp_path / FILE
    f.write_bytes(b"%PDF-1.4 sample tables fixture")
    monkeypatch.setattr(engine.indexing.loader, "load",
                        lambda path, bid_id, content_hash=None, ocr=None: sample_doc(str(path)))
    assert engine.indexing.index_file(f, BID).status == "indexed"
    return engine


def chunk_with(eng, *needles):
    chunks = eng.vector_store.all_chunks()
    return next(c for c in chunks if all(n in c.text for n in needles))


# ----------------------------------------------------------------------- 6. caption-aware chunking
def test_caption_attached_to_its_table_with_page_and_headers():
    chunks = Chunker().chunk(sample_doc())
    t5 = next(c for c in chunks if "Property | 345 | 445 | 222" in c.text and c.page_number == 2)
    assert t5.chunk_type == "table" and t5.page_number == t5.page_end == 2
    lines = t5.text.splitlines()
    assert lines[lines.index("Table 5: year-end financial statement (£, thousands)") + 1] == "2010 | 2009 | 2008"
    assert not any(c.chunk_type == "text" and c.text.endswith("Table 5: year-end financial statement (£, thousands)")
                   for c in chunks)  # no orphan caption chunk
    t4 = next(c for c in chunks if "Daniel Radcliffe" in c.text)
    assert "Table 4: table 3 with column headers added" in t4.text and "Table 5" not in t4.text


def test_merged_captions_attach_only_the_first_and_captionless_tables_stay_unlabelled():
    chunks = Chunker().chunk(sample_doc())
    rain = next(c for c in chunks if "Rainfall" in c.text)
    assert "Table 6: a table with a more serious headings problem" in rain.text and "Table 7" not in rain.text
    assert any(c.chunk_type == "text" and "Table 7: year-end statement" in c.text for c in chunks)
    dup = next(c for c in chunks if c.page_number == 3 and "Property | 345" in c.text)
    assert "Table" not in dup.text.replace("Design and build accessible PDF tables", "")  # never mislabelled
    t8 = next(c for c in chunks if "Cash | 524" in c.text)
    assert t8.text.count("Table 8") == 1 and "Table 7" not in t8.text


def test_ordinary_paragraph_and_other_page_captions_are_not_attached():
    chunks = Chunker().chunk(sample_doc())
    grants = next(c for c in chunks if "Grants | 12" in c.text)
    assert "unaudited" not in grants.text
    assert any(c.chunk_type == "text" and "unaudited" in c.text for c in chunks)
    blocks = [text("Table 9: rainfall by continent", 3), table("A | B", ["x | 1"], 4)]  # caption on another page
    t = next(c for c in Chunker().chunk(sample_doc(blocks=blocks)) if c.chunk_type == "table")
    assert "Table 9" not in t.text


def test_caption_not_duplicated_figure_caption_ignored_and_split_tables_repeat_it():
    already = [text("Table 2: footnotes", 1), table("Table 2: footnotes\nA | B", ["x | 1"], 1)]
    t = next(c for c in Chunker().chunk(sample_doc(blocks=already)) if c.chunk_type == "table")
    assert t.text.count("Table 2: footnotes") == 1
    fig = [text("Figure 3: map of the site", 1), table("A | B", ["x | 1"], 1)]
    t = next(c for c in Chunker().chunk(sample_doc(blocks=fig)) if c.chunk_type == "table")
    assert "Figure 3" not in t.text
    rows = [f"Row {i} | {i * 10}" for i in range(60)]
    big = [text("Table 12: long schedule", 5), table("Name | Value", rows, 5)]
    parts = [c for c in Chunker(max_table_chunk_size=300).chunk(sample_doc(blocks=big)) if c.chunk_type == "table"]
    assert len(parts) > 1 and all("Table 12: long schedule\nName | Value" in p.text for p in parts)


# ----------------------------------------------------------------------- 2. values mapped to years
def test_indexed_table_preserves_row_and_year_relationships(tables_engine):
    t5 = chunk_with(tables_engine, "Table 5:", "Property | 345")
    lines = t5.text.splitlines()
    years = lines[lines.index("2010 | 2009 | 2008")].split(" | ")
    got = {}
    for ln in lines:
        name, *vals = ln.split(" | ")
        if name in EXPECTED:
            got[name] = dict(zip(years, vals))
    assert got == EXPECTED
    assert t5.file_name == FILE and t5.page_number == 2 and t5.bid_id == BID


# ----------------------------------------------------------------------- 1, 7, 11. all retrieval modes
class EvidenceAwareLLM:
    """Answers like a well-behaved model: cites whichever passage is labelled Table 5."""

    def __init__(self, template=ANSWER, finish_reason="stop"):
        self.template, self.finish_reason, self.messages = template, finish_reason, None

    def complete(self, messages, model=None):
        self.messages = messages
        user = messages[1]["content"]
        blocks = re.split(r"\n\n(?=\[\d+\] \()", user.split("Evidence passages:\n\n", 1)[1])
        n = next(int(b[1:b.index("]")]) for b in blocks if "Table 5:" in b and "Property | 345" in b)
        return LLMResponse(text=self.template.format(n=n), model="stub/model:free", latency_s=0.01,
                           usage=LLMUsage(total_tokens=99), finish_reason=self.finish_reason)


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid", "hybrid_rerank"])
def test_table5_question_answered_with_verified_citation_in_every_mode(tables_engine, mode):
    flt = SearchFilters(bid_id=BID)
    hits = tables_engine.retrieval.search(Q, top_k=5, filters=flt, mode=mode)
    assert any("Table 5:" in r.text and "Property | 345" in r.text for r in hits[:3])  # Search still finds it
    qa = QAService(tables_engine.retrieval, EvidenceAwareLLM())
    ans = qa.ask(Q, filters=flt, mode=mode)
    assert ans.status == "verified" and ans.found and ans.error is None, ans.error
    assert ans.ungrounded_values == [] and ans.invalid_citation_refs == []
    (c,) = ans.citations
    assert c.file_name == FILE and c.page_number == 2 and c.bid_id == BID
    assert "Table 5:" in c.text and "Property | 345 | 445 | 222" in c.text
    for row in EXPECTED.values():
        for year, value in row.items():
            assert f"{value} ({year})" in ans.answer
    assert all(r.bid_id == BID for r in ans.evidence)  # filter respected


def test_tokenizer_adds_label_compounds():
    from search.keyword_search import tokenize
    assert "table_5" in tokenize("Table 5: year-end statement")
    assert "clause_7.78" in tokenize("under Clause 7.78 of the agreement")
    assert "addendum_2" in tokenize("Addendum No. 2")
    assert not any("_" in t for t in tokenize("Property | 345 | 445 | 222"))


def test_bm25_ranks_captioned_table_above_captionless_duplicate(tables_engine):
    res = tables_engine.retrieval.search(Q, top_k=10, filters=SearchFilters(bid_id=BID), mode="bm25")
    ids = [r.chunk_id for r in res]
    t5 = chunk_with(tables_engine, "Table 5:", "Property | 345").chunk_id
    dup = next(c.chunk_id for c in tables_engine.vector_store.all_chunks()
               if c.page_number == 3 and "Property | 345" in c.text)
    assert ids[0] == t5 and ids.index(t5) < ids.index(dup)


# ----------------------------------------------------------------------- 3, 4, 5. numeric grounding
def _p(i, txt, page=2):
    return SearchResult(rank=i, score=1.0, chunk_id=f"c{i}", text=txt, bid_id=BID, file_name=FILE,
                        doc_type="other", page_number=page, page_end=page, source_path="/" + FILE,
                        citation=f"{FILE}, p. {page}")


T5_TEXT = "2010 | 2009 | 2008\n" + "\n".join(T5_ROWS[1:4])  # caption-less passage (worst case)


def test_question_label_is_not_an_ungrounded_value():
    ev = [_p(1, T5_TEXT)]
    ans = ANSWER.format(n=1) + "\nTable 5 corresponds to passage 1 [1]."
    assert check_grounding(ans, ev, ev, question=Q) == {"not_in_cited": [], "not_in_evidence": []}
    for q, a in [("What does Figure 2 show for Property?", "Figure 2 shows Property 345 [1]."),
                 ("Under Clause 7.78, what is Property?", "Under Clause 7.78, Property is 345 [1]."),
                 ("What is on page 2?", "Page 2 lists Property 345 [1].")]:
        assert check_grounding(a, ev, ev, question=q)["not_in_evidence"] == [], a


def test_labels_and_question_numbers_are_not_broadly_ignored():
    ev = [_p(1, T5_TEXT)]
    # label not named in the question and absent from the evidence -> still flagged
    assert check_grounding("Table 5 lists Property 345 [1].", ev, ev, question="What is Property?")["not_in_evidence"] == ["5"]
    # a value that merely also occurs in the question is still a claim
    g = check_grounding("Yes, the bid bond is 5% [1].", ev, ev, question="Is the bid bond 5%?")
    assert g["not_in_evidence"] == ["5"]


def test_unsupported_numeric_claim_detected():
    ev = [_p(1, T5_TEXT), _p(2, "Buildings | 346 | 445 | 222", page=9)]
    g = check_grounding(ANSWER.format(n=1).replace("345", "346"), [ev[0]], ev, question=Q)
    assert g["not_in_cited"] == ["346"] and g["not_in_evidence"] == []  # wrong citation
    g = check_grounding(ANSWER.format(n=1).replace("345", "999"), [ev[0]], ev, question=Q)
    assert g["not_in_evidence"] == ["999"]  # hallucinated


@pytest.mark.parametrize("txt,expected", [
    ("pages 2, 3, 9, 10", {"2", "3", "9", "10"}),
    ("pages 2,3,9,10,3,7", {"2", "3", "9", "10", "7"}),
    ("$1,249.00 and 250,000", {"1249", "250000"}),
    ("2010, 2009, and 2008", {"2010", "2009", "2008"}),
    ("12,5", {"12", "5"}),
])
def test_number_extraction_separators(txt, expected):
    assert _numbers(txt) == expected


def test_comma_separated_pages_do_not_create_phantom_numbers():
    ev = [_p(1, T5_TEXT, page=2), _p(2, "Property | 345", page=3), _p(3, "Investment | 567", page=9),
          _p(4, "Intangibles | 423", page=10)]
    g = check_grounding("The tables on pages 2,3,9,10 show Property 345 [1, 2, 3, 4].", ev, ev, question=Q)
    assert g == {"not_in_cited": [], "not_in_evidence": []}


# ----------------------------------------------------------------------- 8. citation validation
class _Fixed:
    def __init__(self, ev):
        self.ev = ev

    def search(self, *a, **k):
        return self.ev


class _Stub:
    def __init__(self, txt, finish_reason="stop"):
        self.txt, self.finish_reason = txt, finish_reason

    def complete(self, messages, model=None):
        return LLMResponse(text=self.txt, model="stub/model:free", latency_s=0.01, usage=LLMUsage(),
                           finish_reason=self.finish_reason)


def _ask(answer, ev=None, **kw):
    ev = ev or [_p(1, "Table 5: year-end\n" + T5_TEXT), _p(2, "Buildings | 345 | 445 | 222", page=9)]
    return QAService(_Fixed(ev), _Stub(answer, **kw)).ask(Q)


@pytest.mark.parametrize("answer,problem", [
    ("Property was 345, 445 and 222.", "no valid citations"),  # missing
    ("Property was 345, 445 and 222 [1, Table 5].", "no valid citations"),  # malformed marker
    ("Property was 345, 445 and 222 (passage 1).", "no valid citations"),  # not the bracket format
    ("Property was 345, 445 and 222 [7].", "cites passages that do not exist"),  # out of range
    ("Investment was 567 [2].", "not found in the cited passages"),  # citation-shaped but wrong
    ("Property was 345 [1].\n\nInvestment was 567, 654 and 423.", "without a citation"),  # partly uncited
])
def test_missing_invalid_or_unsupported_citations_are_flagged(answer, problem):
    ans = _ask(answer)
    assert ans.status == "unverified" and problem in ans.error, ans.error


def test_bullet_list_inherits_intro_citation_and_resolves_to_source():
    ans = _ask("Table 5 values [1]:\n\n- Property: 345 (2010), 445 (2009), 222 (2008)\n"
               "- Investment: 567 (2010), 654 (2009), 423 (2008)")
    assert ans.status == "verified", ans.error
    assert [(c.ref, c.file_name, c.page_number) for c in ans.citations] == [(1, FILE, 2)]


# ----------------------------------------------------------------------- answer layouts seen from models
_I = "In Table 5, the values for Property, Investment, and Intangibles were as follows"
_PV = "345 (2010), 445 (2009), and 222 (2008)"
_IV = "567 (2010), 654 (2009), and 423 (2008)"
_NV = "423 (2010), 123 (2009), and 453 (2008)"
LAYOUTS = {
    "intro_colon_bullets": f"{_I} [{{n}}]:\n\n*   **Property:** {_PV}\n*   **Investment:** {_IV}\n*   **Intangibles:** {_NV}",
    "intro_period_bullets": f"{_I} [{{n}}].\n\n*   **Property:** {_PV}\n*   **Investment:** {_IV}\n*   **Intangibles:** {_NV}",
    "bullets_blank_lines": f"{_I} [{{n}}]:\n\n*   **Property:** {_PV}\n\n*   **Investment:** {_IV}\n\n*   **Intangibles:** {_NV}",
    "bold_groups_bullets": (f"{_I} [{{n}}]:\n\n**Property**\n*   2010: 345\n*   2009: 445\n*   2008: 222\n\n"
                            "**Investment**\n*   2010: 567\n*   2009: 654\n*   2008: 423\n\n"
                            "**Intangibles**\n*   2010: 423\n*   2009: 123\n*   2008: 453"),
    "key_value_lines": f"{_I} [{{n}}]:\n\n**Property:** {_PV}\n**Investment:** {_IV}\n**Intangibles:** {_NV}",
    "trailing_citation": f"{_I}:\n\n*   Property: {_PV}\n*   Investment: {_IV}\n*   Intangibles: {_NV}\n\n[{{n}}]",
    "trailing_source_label": f"{_I}:\n\n*   Property: {_PV}\n*   Investment: {_IV}\n*   Intangibles: {_NV}\n\nSource: [{{n}}]",
    "markdown_heading_table_rows": f"### Table 5 [{{n}}]\n\nItem | 2010 | 2009 | 2008\nProperty | 345 | 445 | 222\n"
                                   "Investment | 567 | 654 | 423\nIntangibles | 423 | 123 | 453",
}


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_equivalent_answer_layouts_verify_identically_in_every_mode(tables_engine, mode, layout):
    qa = QAService(tables_engine.retrieval, EvidenceAwareLLM(template=LAYOUTS[layout]))
    ans = qa.ask(Q, filters=SearchFilters(bid_id=BID), mode=mode)
    assert ans.status == "verified" and ans.error is None, (layout, mode, ans.error)
    assert [(c.file_name, c.page_number) for c in ans.citations] == [(FILE, 2)]


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_inherited_citations_still_check_every_value(layout):
    wrong = LAYOUTS[layout].replace("{n}", "1").replace("654", "656")
    ans = _ask(wrong)
    assert ans.status == "unverified" and "656" in ans.error, (layout, ans.error)


@pytest.mark.parametrize("answer,problem", [
    # a prose paragraph (even an unpunctuated fragment) never inherits a citation
    ("Property was 345 [1].\n\nInvestment was 567, 654 and 423", "without a citation"),
    # prose breaks a structured run: the bullets after it are not covered by the intro
    (f"{_I} [1]:\n\nThe company also reported 999 in other assets.\n\n- Property: {_PV}", "without a citation"),
    # an inherited citation must point at the passage holding the values
    (f"Values [2]:\n\n- Investment: {_IV}", "not found in the cited passages"),
    # a trailing citation covers only the uncited paragraphs directly before it
    (f"Investment: {_IV} [2].\n\n- Property: {_PV}\n\n[1]", "not found in the cited passages"),
])
def test_citation_association_is_not_loosened(answer, problem):
    ans = _ask(answer)
    assert ans.status == "unverified" and problem in ans.error, ans.error


# ----------------------------------------------------------------------- 9. truncated responses
def test_truncated_response_is_incomplete_not_an_answer():
    reasoning = "The user is asking about Table 5. Passage [1] shows Property 345. Passage [2] shows"
    ans = _ask(reasoning, finish_reason="length")
    assert ans.status == "incomplete" and not ans.found and ans.citations == []
    assert "token limit" in ans.error and "max_tokens" in ans.error and ans.finish_reason == "length"


@pytest.mark.parametrize("choice,truncated", [
    ({"finish_reason": "length"}, True),
    ({"finish_reason": "stop", "native_finish_reason": "MAX_TOKENS"}, True),
    ({"native_finish_reason": "max_tokens"}, True),
    ({"finish_reason": "stop"}, False),
    ({}, False),
])
def test_llm_client_preserves_finish_reason(choice, truncated):
    def handler(body, n):
        return httpx.Response(200, json={"model": FREE_A, "choices": [{"message": {"content": "partial"}, **choice}],
                                         "usage": {"completion_tokens": 800}})
    llm, _ = make_llm(handler)
    r = llm.complete([{"role": "user", "content": "q"}])
    assert r.truncated is truncated and r.text == "partial"


def test_unopened_think_block_is_stripped():
    ans = _ask("let me check passage 1 ... 999 ...</think>Property was 345 (2010) [1].")
    assert ans.answer == "Property was 345 (2010) [1]." and ans.status == "verified"


# ----------------------------------------------------------------------- 10. unsupported questions
def test_unsupported_question_returns_not_found_without_facts():
    ans = QAService(_Fixed([_p(1, T5_TEXT)]), _Stub("NOT_FOUND: The information was not found in the indexed documents.")
                    ).ask("In Table 99, what was the Goodwill value for 2015?")
    assert ans.status == "not_found" and not ans.found and ans.citations == []
    assert ans.answer == "The information was not found in the indexed documents."


def test_fabricated_answer_to_unsupported_question_is_flagged():
    q = "In Table 99, what was the Goodwill value for 2015?"
    ev = [_p(1, T5_TEXT)]
    ans = QAService(_Fixed(ev), _Stub("In Table 99, Goodwill was 777 in 2015 [1].")).ask(q)
    assert ans.status == "unverified" and "777" in ans.error and "2015" in ans.error
    assert "99" not in grounding_details("In Table 99, Goodwill was 777 [1].", ev, ev, q)["not_in_evidence"]
