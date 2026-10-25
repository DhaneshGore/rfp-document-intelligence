"""Hybrid retrieval: dense (Chroma) + BM25, fused with Reciprocal Rank Fusion,
optionally re-ranked by a cross-encoder.

Modes (used by the API, UI and evaluation):
  vector         dense only
  bm25           keyword only
  hybrid         dense + BM25 fused with weighted RRF
  hybrid_rerank  hybrid, then cross-encoder re-ranking of the top candidates
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import get_settings
from search.keyword_search import BM25Index
from search.query_expansion import expand_query, has_change_intent
from search.reranker import Reranker
from search.vector_store import ChromaVectorStore
from schemas import Chunk, Citation, SearchFilters, SearchResult

log = logging.getLogger(__name__)

SEARCH_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class _Candidate:
    chunk: Chunk
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


def rrf_fuse(
    ranked_lists: dict[str, list[tuple[Chunk, float]]],
    k: int = 60,
    weights: Optional[dict[str, float]] = None,
) -> list[_Candidate]:
    """Weighted Reciprocal Rank Fusion: score = sum_i w_i / (k + rank_i)."""
    weights = weights or {}
    cands: dict[str, _Candidate] = {}
    for name, results in ranked_lists.items():
        for rank, (chunk, score) in enumerate(results, start=1):
            c = cands.setdefault(chunk.chunk_id, _Candidate(chunk))
            c.scores[name] = score
            c.ranks[name] = rank
            c.scores["rrf"] = c.scores.get("rrf", 0.0) + weights.get(name, 1.0) / (k + rank)
    return sorted(cands.values(), key=lambda c: c.scores["rrf"], reverse=True)


def to_result(c: _Candidate, rank: int, score_key: str) -> SearchResult:
    ch = c.chunk
    cit = Citation(
        file_name=ch.file_name,
        page_number=ch.page_number,
        page_end=ch.page_end,
        bid_id=ch.bid_id,
        doc_type=ch.doc_type.value,
        chunk_id=ch.chunk_id,
        source_path=ch.source_path,
        section=ch.section,
    )
    return SearchResult(
        rank=rank,
        score=round(c.scores.get(score_key, 0.0), 6),
        chunk_id=ch.chunk_id,
        text=ch.text,
        bid_id=ch.bid_id,
        file_name=ch.file_name,
        doc_type=ch.doc_type.value,
        page_number=ch.page_number,
        page_end=ch.page_end,
        addendum_number=ch.addendum_number,
        document_date=ch.document_date,
        section=ch.section,
        source_path=ch.source_path,
        citation=cit.label,
        scores={k: round(v, 6) for k, v in c.scores.items()},
        ranks=dict(c.ranks),
    )


class HybridSearcher:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_provider: Callable[[], BM25Index],
        reranker: Optional[Reranker] = None,
    ):
        self.vector_store = vector_store
        self.bm25_provider = bm25_provider
        self.reranker = reranker
        self.settings = get_settings()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        mode: Optional[str] = None,
        expand: Optional[bool] = None,
    ) -> list[SearchResult]:
        s = self.settings
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must not be empty")
        mode = mode or s.default_search_mode
        if mode not in SEARCH_MODES:
            raise ValueError(f"Unknown search mode '{mode}'. Use one of {SEARCH_MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            log.info("Reranker disabled; falling back to plain hybrid")
            mode = "hybrid"
        top_k = top_k or s.default_top_k
        pool = max(s.candidate_pool, top_k)
        expand = s.query_expansion if expand is None else expand
        kw_query = expand_query(query) if expand else query

        lists: dict[str, list[tuple[Chunk, float]]] = {}
        if mode != "bm25":
            lists["dense"] = self.vector_store.query(query, pool, filters)
        if mode != "vector":
            lists["bm25"] = self.bm25_provider().search(kw_query, pool, filters)

        if mode in ("vector", "bm25"):
            key = "dense" if mode == "vector" else "bm25"
            cands = rrf_fuse({key: lists[key]}, k=s.rrf_k)
            score_key = key
        else:
            weights = {"dense": s.dense_weight, "bm25": s.bm25_weight}
            # Addendums override the base RFP: for "revised / latest / after addendum"
            # questions, addendum chunks found by either retriever get an extra vote.
            if s.addendum_intent_weight > 0 and has_change_intent(query):
                fused = rrf_fuse(lists, k=s.rrf_k)
                addenda = [(c.chunk, c.scores["rrf"]) for c in fused if c.chunk.doc_type.value == "addendum"]
                if addenda:
                    lists["addendum_intent"] = addenda
                    weights["addendum_intent"] = s.addendum_intent_weight
            cands = rrf_fuse(lists, k=s.rrf_k, weights=weights)
            score_key = "rrf"

        if mode == "hybrid_rerank" and cands:
            head = cands[: max(s.rerank_candidates, top_k)]
            for c, sc in zip(head, self.reranker.score(query, [c.chunk.text for c in head])):
                c.scores["rerank"] = sc
            by_rerank = sorted(head, key=lambda c: c.scores["rerank"], reverse=True)
            if s.rerank_fusion == "rrf":
                # The cross-encoder is one vote alongside the hybrid ranking, so a strong
                # hybrid candidate cannot be discarded by a single reranker misjudgement.
                rr_rank = {id(c): i for i, c in enumerate(by_rerank, start=1)}
                for i, c in enumerate(head, start=1):
                    c.ranks["rerank"] = rr_rank[id(c)]
                    c.scores["final"] = 1 / (s.rrf_k + i) + s.rerank_weight / (s.rrf_k + rr_rank[id(c)])
                cands = sorted(head, key=lambda c: c.scores["final"], reverse=True)
                score_key = "final"
            else:
                cands = by_rerank
                score_key = "rerank"

        return [to_result(c, i, score_key) for i, c in enumerate(cands[:top_k], start=1)]
