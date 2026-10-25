import pytest

from search.hybrid_search import HybridSearcher, rrf_fuse
from search.keyword_search import BM25Index, tokenize
from search.query_expansion import expand_query
from schemas import Chunk, DocType, SearchFilters
from tests.conftest import ReverseReranker


@pytest.fixture
def indexed(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    return engine


def _chunk(cid, text, **kw):
    base = dict(chunk_id=cid, doc_id="d", text=text, bid_id="B", file_name="f.pdf",
                source_path="/f.pdf", doc_type=DocType.RFP)
    base.update(kw)
    return Chunk(**base)


def test_tokenizer_keeps_identifiers():
    toks = tokenize("Bid JA-207652 needs SKU 210-BLYZ and the WD22TB4 dock.")
    assert "ja-207652" in toks and "207652" in toks
    assert "210-blyz" in toks and "blyz" in toks
    assert "wd22tb4" in toks
    assert "the" not in toks and "and" not in toks


def test_query_expansion():
    q = expand_query("What is the deadline?")
    assert "due date" in q and "closing date" in q
    assert expand_query("unrelated words") == "unrelated words"


def test_rrf_fusion_math():
    a, b, c = _chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")
    fused = rrf_fuse({"dense": [(a, 0.9), (b, 0.8)], "bm25": [(b, 5.0), (c, 3.0)]}, k=60)
    assert [x.chunk.chunk_id for x in fused][0] == "b"  # appears in both lists
    assert fused[0].scores["rrf"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].ranks == {"dense": 2, "bm25": 1}


def test_bm25_respects_filters():
    idx = BM25Index([_chunk("1", "bid bond required", bid_id="B1"), _chunk("2", "bid bond required", bid_id="B2")])
    res = idx.search("bid bond", 10, SearchFilters(bid_id="B2"))
    assert [c.chunk_id for c, _ in res] == ["2"]


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_modes_return_cited_results(indexed, mode):
    res = indexed.retrieval.search("bid bond amount", top_k=3, mode=mode)
    assert res and res[0].rank == 1
    for r in res:
        assert r.file_name and r.bid_id == "BidA" and r.chunk_id and r.doc_type
        assert r.citation.startswith(r.file_name)
        if r.page_number is not None:
            assert f"p. {r.page_number}" in r.citation or f"pp. {r.page_number}" in r.citation


def test_hybrid_finds_exact_identifier(indexed):
    res = indexed.retrieval.search("210-ABCD", top_k=3, mode="hybrid")
    assert res[0].file_name == "Widget_Specs.pdf"
    assert "bm25" in res[0].scores


def test_hybrid_uses_both_retrievers(indexed):
    res = indexed.retrieval.search("bid bond percentage", top_k=5, mode="hybrid")
    assert any("dense" in r.ranks for r in res)
    assert any("bm25" in r.ranks for r in res)
    top = res[0]
    assert "5%" in top.text and top.page_number <= 2 <= top.page_end


def test_metadata_filters(indexed):
    r = indexed.retrieval
    assert {x.doc_type for x in r.search("due date", filters=SearchFilters(doc_type="addendum"))} == {"addendum"}
    assert all(x.addendum_number == 1 for x in r.search("due date", filters=SearchFilters(addendum_number=1)))
    assert r.search("due date", filters=SearchFilters(bid_id="NoSuchBid")) == []
    specs = r.search("memory", filters=SearchFilters(bid_id="BidA", doc_type="specs"), mode="hybrid")
    assert specs and {x.file_name for x in specs} == {"Widget_Specs.pdf"}


def test_html_results_have_no_page_number(indexed):
    res = indexed.retrieval.search("closing date", filters=SearchFilters(doc_type="bid_page"), mode="bm25")
    assert res and res[0].page_number is None
    cit = res[0].citation
    assert cit.startswith(res[0].file_name) and "p. " not in cit  # never invent a page number
    if res[0].section:  # page-less passages are located by their section heading instead
        assert cit == f'{res[0].file_name}, section "{res[0].section.split(" > ")[-1]}"'


def test_reranker_reorders(indexed):
    searcher = HybridSearcher(indexed.vector_store, indexed.retrieval._get_bm25, ReverseReranker())
    res = searcher.search("warranty", top_k=5, mode="hybrid_rerank")
    scores = [r.scores["rerank"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert all(r.score == r.scores["rerank"] for r in res)


def test_rerank_mode_without_reranker_falls_back(indexed):
    res = indexed.retrieval.search("warranty", mode="hybrid_rerank")
    assert res and "rerank" not in res[0].scores


def test_invalid_inputs(indexed):
    with pytest.raises(ValueError):
        indexed.retrieval.search("   ")
    with pytest.raises(ValueError):
        indexed.retrieval.search("x", mode="magic")


def test_empty_index_returns_nothing(engine):
    assert engine.retrieval.search("anything", mode="hybrid") == []


def test_bm25_refreshes_after_new_document(indexed, bid_folder):
    from tests.conftest import make_pdf

    assert indexed.retrieval.search("zebrafish", mode="bm25") == []
    make_pdf(bid_folder / "Addendum 3.pdf", ["ADDENDUM No. 3\nZebrafish compliance form is now required."])
    indexed.indexing.index_folder(bid_folder)
    res = indexed.retrieval.search("zebrafish", mode="bm25")
    assert res and res[0].file_name == "Addendum 3.pdf" and res[0].addendum_number == 3
