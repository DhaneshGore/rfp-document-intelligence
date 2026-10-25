"""End-to-end answer evaluation against the running API (real OpenRouter calls).

    python main.py serve                      # in another terminal (LLM_MAX_RETRIES=0 recommended)
    python -m eval.llm_e2e                    # default model, all questions
    python -m eval.llm_e2e --compare-models   # also 2 questions on every other configured model

Each answer is checked automatically against facts located manually in the documents:
required phrases, the expected cited file/page, the bid filter, and the numeric
grounding check from services.qa_service. Requests are spaced out to respect free-tier
limits and are never retried here. Results go to eval/results/llm_e2e.json and .md.
The API key is never printed.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

import config  # noqa: F401
from config import get_settings

HERE = Path(__file__).resolve().parent

QUESTIONS = [
    {
        "id": "Q1", "bid_id": "Bid1",
        "question": "What is the revised submission deadline in Bid1?",
        "must_include": [["july 9, 2024", "07/09/2024"], ["2:00 pm", "2 pm"]],
        "expected_sources": [{"file": "Addendum 2", "page": 1}],
    },
    {
        "id": "Q2", "bid_id": "Bid1",
        "question": "What was the original submission deadline before the addendum?",
        "must_include": [["27-jun-2024", "june 27, 2024", "06/27/2024", "27 june 2024"]],
        "expected_sources": [{"file": "FINAL.pdf", "page": 2}],
    },
    {
        "id": "Q3", "bid_id": "Bid2",
        "question": "What processor, memory, and storage are specified in the Dell laptop specification document?",
        "must_include": [["core ultra 5 125u"], ["16 gb"], ["256 gb"]],
        "expected_sources": [{"file": "Dell_Laptop_Specs", "page": 1}],
    },
    {
        "id": "Q4", "bid_id": "Bid1",
        "question": "What is the initial contract term and the maximum possible term in the main RFP?",
        "must_include": [["three (3) year", "3-year", "3 year", "three year", "three-year"],
                         ["5 years", "five (5) years", "five years", "5 year"]],
        "expected_sources": [{"file": "FINAL.pdf", "page": 2}],
    },
    {
        "id": "Q5", "bid_id": "Bid2",
        "question": "What does the Contract Affidavit say about financial disclosure?",
        "must_include": [["$100,000", "100,000"], ["30 days", "thirty"]],
        "expected_sources": [{"file": "Contract_Affidavit", "page": 1}],
    },
    {
        "id": "Q6", "bid_id": "Bid2",
        "question": "What is the minimum general liability insurance coverage amount required for the Dell laptop bid?",
        "expect_not_found": True,
        "expected_sources": [],
    },
    {
        "id": "Q7", "bid_id": "Bid2",
        "question": "What is the submission deadline?",
        "must_include": [["06/10/2024", "june 10, 2024"]],
        "must_exclude": ["july 9"],
        "expected_sources": [{"file": "PORFP", "page": 1}, {"file": "BidNet Direct.html", "page": None}],
    },
]
COMPARE_IDS = ["Q1", "Q3"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def assess(q: dict, ans: dict) -> dict:
    text = _norm(ans.get("answer") or "")
    cits = ans.get("citations") or []
    out: dict = {
        "llm_error": ans.get("error") if not ans.get("answer") else None,
        "found": ans.get("found"),
        "citations": [c["citation"] for c in cits],
        "invalid_refs": ans.get("invalid_citation_refs") or [],
        "ungrounded_values": ans.get("ungrounded_values") or [],
        "wrong_bid_citations": [c["citation"] for c in cits if c["bid_id"] != q["bid_id"]],
    }
    if out["llm_error"]:
        out["verdict"] = "error"
        return out
    if q.get("expect_not_found"):
        hedged = any(p in text for p in ("not found", "not stated", "not specified", "does not", "no information", "not mention", "not contain", "not provide"))
        out["verdict"] = "correct" if (not ans.get("found") or hedged) and not out["ungrounded_values"] else "incorrect"
        return out
    missing = [alts[0] for alts in q.get("must_include", []) if not any(a in text for a in alts)]
    excluded = [x for x in q.get("must_exclude", []) if x in text]
    src_ok = any(
        any(e["file"].lower() in c["file_name"].lower() and (e["page"] is None or
            (c.get("page_number") is not None and c["page_number"] <= e["page"] <= (c.get("page_end") or c["page_number"])))
            for c in cits)
        for e in q["expected_sources"]
    )
    out.update(missing_facts=missing, forbidden_present=excluded, expected_source_cited=src_ok)
    ok = ans.get("found") and not missing and not excluded and src_ok and cits \
        and not out["invalid_refs"] and not out["wrong_bid_citations"]
    out["verdict"] = "correct" if ok and not out["ungrounded_values"] else ("partial" if ans.get("found") and cits else "incorrect")
    return out


def ask(api: str, q: dict, model: str, timeout: float) -> tuple[dict, float]:
    t = time.time()
    r = httpx.post(f"{api}/ask", json={"question": q["question"], "bid_id": q["bid_id"], "model": model},
                   timeout=timeout)
    wall = time.time() - t
    r.raise_for_status()
    return r.json(), wall


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-models", action="store_true")
    p.add_argument("--delay", type=float, default=4.0, help="seconds between requests (free-tier friendly)")
    p.add_argument("--only", nargs="*", help="question ids to run on the default model")
    p.add_argument("--model", help="run the main questions on this configured model instead of the default")
    args = p.parse_args()
    s = get_settings()
    api = s.api_url.rstrip("/")
    models_info = httpx.get(f"{api}/models", timeout=30).json()
    status = {m["id"]: m["status"] for m in models_info["models"]}
    default = args.model or models_info["default"]

    runs = []
    plan = [(default, q) for q in QUESTIONS if not args.only or q["id"] in args.only]
    if args.compare_models:
        for m in models_info["models"]:
            if m["id"] != default:
                plan += [(m["id"], q) for q in QUESTIONS if q["id"] in COMPARE_IDS]
    rate_limited: set[str] = set()
    for i, (model, q) in enumerate(plan):
        if model in rate_limited:  # do not keep hitting a rate-limited free model
            runs.append({"model": model, "catalogue_status": status.get(model), "qid": q["id"], "bid_id": q["bid_id"],
                         "question": q["question"], "answer": None, "llm_latency_s": None, "wall_s": 0,
                         "usage": None, "served_by": None, "verdict": "skipped (model rate-limited)",
                         "citations": [], "llm_error": "not sent: model was rate-limited earlier in this run"})
            print(f"[{model}] {q['id']}: skipped (rate-limited earlier)", flush=True)
            continue
        if i:
            time.sleep(args.delay)
        try:
            ans, wall = ask(api, q, model, timeout=s.llm_timeout_s * 3)
        except httpx.HTTPError as exc:
            ans, wall = {"error": f"HTTP error: {exc}", "answer": ""}, 0.0
        res = assess(q, ans)
        if "RateLimit" in str(ans.get("error") or ""):
            rate_limited.add(model)
        runs.append({
            "model": model, "catalogue_status": status.get(model), "qid": q["id"], "bid_id": q["bid_id"],
            "question": q["question"], "answer": ans.get("answer"), "llm_latency_s": ans.get("latency_s"),
            "wall_s": round(wall, 2), "usage": ans.get("usage"), "served_by": ans.get("model"), **res,
        })
        print(f"[{model}] {q['id']}: {res['verdict']}  cites={res.get('citations')}  "
              f"latency={ans.get('latency_s')}  err={res.get('llm_error') or ''}", flush=True)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    (out_dir / "llm_e2e.json").write_text(json.dumps({"timestamp": stamp, "runs": runs}, indent=2), encoding="utf-8")
    lines = [f"Run: {stamp}", "", "| Model | Q | Verdict | Cited | LLM latency (s) | Tokens | Cost | Error |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        u = r.get("usage") or {}
        lines.append(f"| {r['model']} | {r['qid']} | {r['verdict']} | {'; '.join(r['citations'] or []) or '-'} | "
                     f"{r['llm_latency_s'] if r['llm_latency_s'] is not None else '-'} | {u.get('total_tokens', '-')} | "
                     f"{u.get('cost', '-')} | {(r['llm_error'] or '')[:80]} |")
    (out_dir / "llm_e2e.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
