"""Retrieval evaluation: Recall@k, MRR@10 and nDCG@10 for several configurations.

    python -m eval.evaluate                      # bid-filtered (default), all modes
    python -m eval.evaluate --no-filter          # search across all bids
    python -m eval.evaluate --modes vector hybrid

Relevance is judged from the labels in eval/questions.json (file + page + phrase),
so results do not depend on chunk ids and stay valid if chunking changes.
Results are written to eval/results/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import config  # noqa: F401
from schemas import Chunk, SearchFilters, SearchResult

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5, 10)

CONFIGS = {
    "vector": {"mode": "vector", "expand": False},
    "bm25": {"mode": "bm25", "expand": True},
    "hybrid_no_expansion": {"mode": "hybrid", "expand": False},
    "hybrid": {"mode": "hybrid", "expand": True},
    "hybrid_rerank": {"mode": "hybrid_rerank", "expand": True},
    # variants (setting overrides applied only for that configuration)
    "hybrid_intent": {"mode": "hybrid", "expand": True, "overrides": {"addendum_intent_weight": 1.0}},
    "hybrid_rerank_rrf": {"mode": "hybrid_rerank", "expand": True, "overrides": {"rerank_fusion": "rrf"}},
    "hybrid_rerank_rrf_intent": {"mode": "hybrid_rerank", "expand": True,
                                 "overrides": {"rerank_fusion": "rrf", "addendum_intent_weight": 1.0}},
}
BASELINE_OVERRIDES = {"rerank_fusion": "replace", "addendum_intent_weight": 0.0}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def is_relevant(item: Chunk | SearchResult, labels: list[dict]) -> bool:
    text = _norm(item.text)
    for lab in labels:
        if lab["file"].lower() not in item.file_name.lower():
            continue
        pages = lab.get("pages")
        if pages and item.page_number is not None:
            lo, hi = item.page_number, item.page_end or item.page_number
            if not any(lo <= p <= hi for p in pages):
                continue
        if any(_norm(c) in text for c in lab["contains"]):
            return True
    return False


def metrics_for(rels: list[bool], n_relevant: int) -> dict[str, float]:
    out = {f"recall@{k}": float(any(rels[:k])) for k in KS}
    first = next((i for i, r in enumerate(rels[:10], start=1) if r), None)
    out["mrr@10"] = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:10], start=1) if r)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def run(modes: list[str], use_filter: bool, questions_path: Path) -> dict:
    from services.container import get_engine

    eng = get_engine()
    qs = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    corpus = eng.vector_store.all_chunks()

    # Sanity check: every question must have >=1 relevant chunk in the index.
    n_rel = {}
    for q in qs:
        n_rel[q["id"]] = sum(
            1 for c in corpus
            if is_relevant(c, q["relevant"]) and (not use_filter or c.bid_id == q["bid_id"])
        )
    unanswerable = [qid for qid, n in n_rel.items() if n == 0]

    report: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bid_filter": use_filter,
        "num_questions": len(qs),
        "corpus_chunks": len(corpus),
        "questions_without_relevant_chunks": unanswerable,
        "settings": {
            "embedding_model": eng.settings.embedding_model,
            "reranker_model": eng.settings.reranker_model,
            "chunk_size": eng.settings.chunk_size,
            "chunk_overlap": eng.settings.chunk_overlap,
            "candidate_pool": eng.settings.candidate_pool,
            "rrf_k": eng.settings.rrf_k,
        },
        "configs": {},
    }
    for name in modes:
        cfg = CONFIGS[name]
        # every configuration starts from the same baseline, plus its own overrides
        for key, val in {**BASELINE_OVERRIDES, **cfg.get("overrides", {})}.items():
            setattr(eng.settings, key, val)
        per_q, latencies = [], []
        for q in qs:
            filters = SearchFilters(bid_id=q["bid_id"]) if use_filter else None
            t = time.time()
            results = eng.retrieval.search(q["question"], top_k=10, filters=filters, mode=cfg["mode"], expand=cfg["expand"])
            latencies.append(time.time() - t)
            rels = [is_relevant(r, q["relevant"]) for r in results]
            m = metrics_for(rels, n_rel[q["id"]])
            first = next((i for i, r in enumerate(rels, start=1) if r), None)
            per_q.append({"id": q["id"], "first_relevant_rank": first, **m,
                          "top3": [r.citation for r in results[:3]]})
        agg = {k: round(sum(p[k] for p in per_q) / len(per_q), 4) for k in per_q[0] if k.startswith(("recall", "mrr", "ndcg"))}
        agg["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
        report["configs"][name] = {"aggregate": agg, "per_question": per_q}
    return report


def to_markdown(report: dict) -> str:
    cols = [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10", "avg_latency_s"]
    lines = [
        f"Questions: {report['num_questions']} · bid filter: {'on' if report['bid_filter'] else 'off'} · "
        f"corpus: {report['corpus_chunks']} chunks · {report['timestamp']}",
        "",
        "| Configuration | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for name, data in report["configs"].items():
        a = data["aggregate"]
        lines.append(f"| {name} | " + " | ".join(f"{a[c]:.3f}" for c in cols) + " |")
    if report["questions_without_relevant_chunks"]:
        lines.append(f"\nQuestions with no relevant chunk in index: {report['questions_without_relevant_chunks']}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--no-filter", action="store_true", help="do not restrict search to the question's bid")
    p.add_argument("--questions", type=Path, default=HERE / "questions.json")
    args = p.parse_args()
    report = run(args.modes, not args.no_filter, args.questions)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    tag = "filtered" if report["bid_filter"] else "unfiltered"
    (out_dir / f"retrieval_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(report)
    (out_dir / f"retrieval_{tag}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    for name, data in report["configs"].items():
        misses = [p["id"] for p in data["per_question"] if not p["first_relevant_rank"]]
        if misses:
            print(f"{name}: no relevant result in top-10 for {misses}")


if __name__ == "__main__":
    main()
