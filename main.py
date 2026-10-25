"""Command-line entry point.

    python main.py index --all                 # index every bid folder (incremental)
    python main.py index --bid ./Bid1          # index one folder (bid id = folder name)
    python main.py index --all --rebuild       # wipe and rebuild the index
    python main.py search "due date" --bid-id Bid1 --top-k 5 [--mode hybrid_rerank]
    python main.py ask "Is a bid bond required?" --bid-id Bid1 [--model ...]
    python main.py extract --bid Bid1          # 20 Section 8 fields -> output/Bid1.json + trace
    python main.py extract --bid ./NewBid      # index a new folder (incremental), then extract
    python main.py models                      # show configured models + free verification
    python main.py status                      # indexed bids / documents
    python main.py serve                       # FastAPI on API_HOST:API_PORT
    python main.py ui                          # Streamlit UI
    python main.py up                          # API + UI together
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config  # noqa: F401  (sets USE_TF before heavy imports)
from config import PROJECT_ROOT, get_settings


def _filters(args):
    from schemas import SearchFilters

    return SearchFilters(
        bid_id=args.bid_id, doc_type=args.doc_type, addendum_number=args.addendum_number,
    )


def _print_progress(i: int, n: int, name: str) -> None:
    if i < n:
        print(f"  [{i + 1}/{n}] {name} ...", flush=True)


def cmd_index(args) -> int:
    from services.container import get_engine

    eng = get_engine()
    if args.rebuild:
        report = eng.indexing.rebuild(progress=_print_progress)
    elif args.all:
        report = eng.indexing.index_all(force=args.force, progress=_print_progress)
    elif args.bid:
        report = eng.indexing.index_folder(
            Path(args.bid), bid_id=args.bid_id, force=args.force, progress=_print_progress
        )
    else:
        print("Specify --all, --bid PATH or --rebuild", file=sys.stderr)
        return 2
    for f in report.files:
        warn = f"  warnings: {f.warnings}" if f.warnings else ""
        print(f"{f.status:18} {f.bid_id:8} {f.doc_type or '-':10} {f.chunks:4} chunks {f.seconds:6.1f}s  {f.file_name}  {f.message}{warn}")
    print(f"Summary: {report.summary}  chunks added: {report.total_chunks_added}  total time: {report.seconds}s")
    return 1 if any(f.status == "failed" for f in report.files) else 0


def cmd_search(args) -> int:
    from services.container import get_engine

    eng = get_engine()
    t = time.time()
    results = eng.retrieval.search(
        args.query, top_k=args.top_k, filters=_filters(args), mode=args.mode, expand=not args.no_expand
    )
    if args.json:
        print(json.dumps([r.model_dump() for r in results], indent=2))
        return 0
    print(f"{len(results)} results in {time.time() - t:.2f}s")
    for r in results:
        print(f"\n#{r.rank} score={r.score:.4f} [{r.citation}] bid={r.bid_id} type={r.doc_type} chunk={r.chunk_id}")
        print("   " + r.text[:400].replace("\n", "\n   "))
    return 0


def cmd_ask(args) -> int:
    from agents import Orchestrator
    from extraction.trace import ExtractionTrace
    from services.container import get_engine

    # Question mode: Retrieval agent -> Q&A agent (same answer, citations and checks as /ask)
    trace = ExtractionTrace(args.bid_id or "all-bids")
    state = Orchestrator.from_engine(get_engine(), model=args.model).run_qa(
        args.question, filters=_filters(args), mode=args.mode, model=args.model, trace=trace)
    if args.trace:
        trace.write_jsonl(Path(args.trace))
    ans = state.answer
    if args.json:
        print(ans.model_dump_json(indent=2))
        return 0
    print(f"status: {ans.status}" + (f" (finish_reason={ans.finish_reason})" if ans.finish_reason else ""))
    if ans.error:
        print(f"WARNING/ERROR: {ans.error}")
    print(f"\n{ans.answer}\n")
    for c in ans.citations:
        print(f"  [{c.ref}] {c.citation} (bid {c.bid_id}, {c.doc_type}, chunk {c.chunk_id})")
    if ans.model:
        print(f"\nmodel={ans.model} latency={ans.latency_s}s usage={ans.usage}")
    return 0 if not ans.error else 1


def cmd_extract(args) -> int:
    """Extract the 20 Section 8 fields for one bid -> <out>/<bid>.json + <bid>.trace.jsonl.

    --bid is an indexed bid id, or a bid folder path (indexed incrementally first).
    Exit code: 0 written, 1 written but some fields hit LLM/format errors, 2 not run.
    """
    from agents import BidNotIndexedError, Orchestrator
    from extraction.service import write_extraction
    from extraction.trace import ExtractionTrace
    from services.container import get_engine
    from services.llm_service import OpenRouterLLM
    from services.uploads import UploadError, validate_bid_id

    eng = get_engine()
    if isinstance(eng.llm, OpenRouterLLM) and not eng.settings.openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY is not set; extraction needs the LLM.", file=sys.stderr)
        return 2
    folder = Path(args.bid)
    try:
        bid_id = validate_bid_id(args.bid_id or folder.resolve().name) if folder.is_dir() else validate_bid_id(args.bid)
    except UploadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if folder.is_dir():
        print(f"Indexing {folder} as bid '{bid_id}' (unchanged files are skipped) ...", flush=True)

    orch = Orchestrator.from_engine(eng, model=args.model, max_retries=args.max_retries,
                                    retry_not_found=args.retry_not_found, reconcile=not args.no_reconcile,
                                    workers=args.workers)
    trace = ExtractionTrace(bid_id)
    try:
        state = orch.run_extraction(bid_id=bid_id, folder=str(folder) if folder.is_dir() else None, trace=trace)
    except BidNotIndexedError as exc:
        print(f"ERROR: {exc}. Pass a bid folder path or run `python main.py index` first.", file=sys.stderr)
        return 2
    record = state.final
    if record is None:
        print(f"ERROR: the run produced no record: {state.errors}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    path = write_extraction(record, out_dir)
    tpath = trace.write_jsonl(out_dir / f"{bid_id}.trace.jsonl")
    rpath = out_dir / f"{bid_id}.report.md"
    rpath.write_text(redact_text(state.report_markdown or ""), encoding="utf-8")
    for name, f in record.fields.items():
        value = f.value if not isinstance(f.value, list) else "; ".join(f.value)
        print(f"{f.status:11} {f.confidence:4.2f}  {name:40} {str(value)[:70] if value is not None else '-'}")
    v = record.validation
    print(f"\npassed={v.passed} failed={v.failed} (unverified={v.unverified}, error={v.error}) "
          f"not_found={v.not_found}  addendum_changes={len(record.addendum_changes)}  llm_calls={record.llm_calls}")
    print(f"JSON:   {path}\ntrace:  {tpath}\nreport: {rpath}")
    if state.errors:
        print(f"agent errors: {state.errors}", file=sys.stderr)
    return 1 if v.error or state.errors else 0


def redact_text(text: str) -> str:
    from services.llm_service import redact

    return redact(text)


def cmd_models(args) -> int:
    from services.llm_service import OpenRouterLLM

    for m in OpenRouterLLM().model_statuses():
        flags = ("default " if m["default"] else "") + ("fallback" if m["fallback"] else "")
        print(f"{m['status']:14} {m['id']:45} {flags:16} {m['detail']}")
    return 0


def cmd_status(args) -> int:
    from services.container import get_engine

    eng = get_engine()
    print(json.dumps(eng.retrieval.stats(), indent=2))
    for d in eng.retrieval.documents():
        print(f"{d['status']:18} {d['bid_id']:8} {d.get('doc_type') or '-':10} {d.get('chunks', 0):4}  {d['file_name']}")
    return 0


UI_PORT = 8501


def _streamlit_cmd() -> list[str]:
    s = get_settings()
    return [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py"),
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
            "--server.port", str(UI_PORT), "--server.maxUploadSize", str(s.max_upload_mb)]


def _api_cmd() -> list[str]:
    # Access log off: api.routes logs every request itself, with its duration.
    s = get_settings()
    return [sys.executable, "-m", "uvicorn", "api.routes:app", "--host", s.api_host, "--port", str(s.api_port),
            "--log-level", s.log_level.lower(), "--no-access-log"]


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def cmd_serve(args) -> int:
    import uvicorn

    s = get_settings()
    uvicorn.run("api.routes:app", host=s.api_host, port=s.api_port, log_level=s.log_level.lower(),
                access_log=False)
    return 0


def cmd_ui(args) -> int:
    return subprocess.call(_streamlit_cmd(), cwd=PROJECT_ROOT)


def cmd_up(args) -> int:
    """Run API + UI in this terminal; their output is prefixed [api] / [ui]. If either
    exits, the other is stopped too, so a crashed API is never hidden behind a running UI."""
    import os
    import threading

    s = get_settings()
    busy = [f"{name} port {port}" for name, host, port in
            (("API", s.api_host, s.api_port), ("UI", "127.0.0.1", UI_PORT)) if _port_in_use(host, port)]
    if busy:
        print(f"ERROR: {' and '.join(busy)} already in use - another API/UI is still running.\n"
              "Stop it first (close its terminal or end the python process), or set API_PORT in .env.",
              file=sys.stderr)
        return 1

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    procs: dict[str, subprocess.Popen] = {}

    def pump(name: str, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            print(f"[{name}] {line.rstrip()}", flush=True)

    for name, cmd in (("api", _api_cmd()), ("ui", _streamlit_cmd())):
        procs[name] = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=pump, args=(name, procs[name]), daemon=True).start()
    print(f"API: http://{s.api_host}:{s.api_port}/docs   UI: http://localhost:{UI_PORT}   "
          f"(log level {s.log_level}; Ctrl+C stops both)", flush=True)
    code = 0
    try:
        while True:
            exited = [(n, p.returncode) for n, p in procs.items() if p.poll() is not None]
            if exited:
                name, code = exited[0]
                print(f"ERROR: {name.upper()} process exited with code {code}; stopping the other one.",
                      file=sys.stderr, flush=True)
                code = code or 1
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping API and UI ...", flush=True)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RFP Intelligence Platform - RAG search engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index bid folders (incremental)")
    pi.add_argument("--all", action="store_true", help="index every bid folder + uploads")
    pi.add_argument("--bid", help="path to one bid folder")
    pi.add_argument("--bid-id", help="override bid id (default: folder name)")
    pi.add_argument("--force", action="store_true", help="re-process even unchanged files")
    pi.add_argument("--rebuild", action="store_true", help="wipe the index and rebuild everything")
    pi.set_defaults(fn=cmd_index)

    for name, fn in (("search", cmd_search), ("ask", cmd_ask)):
        sp = sub.add_parser(name)
        sp.add_argument("query" if name == "search" else "question")
        sp.add_argument("--bid-id")
        sp.add_argument("--doc-type")
        sp.add_argument("--addendum-number", type=int)
        sp.add_argument("--mode", choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
        sp.add_argument("--json", action="store_true")
        if name == "search":
            sp.add_argument("--top-k", type=int, default=5)
            sp.add_argument("--no-expand", action="store_true", help="disable query expansion")
        else:
            sp.add_argument("--model")
            sp.add_argument("--trace", help="write the agent trace (JSON lines) to this file")
        sp.set_defaults(fn=fn)

    pe = sub.add_parser("extract", help="extract the 20 Section 8 fields for one bid (JSON + trace)")
    pe.add_argument("--bid", required=True, help="indexed bid id, or path to a bid folder")
    pe.add_argument("--bid-id", help="bid id to use when --bid is a folder (default: folder name)")
    pe.add_argument("--out", default=str(PROJECT_ROOT / "output"), help="output directory")
    pe.add_argument("--model", help="configured free OpenRouter model (default: LLM_DEFAULT_MODEL)")
    pe.add_argument("--max-retries", type=int, default=1, help="retry rounds for failed fields (default 1)")
    pe.add_argument("--retry-not-found", action="store_true",
                    help="also retry not-found fields when a wider search finds new passages")
    pe.add_argument("--no-reconcile", action="store_true", help="skip addendum reconciliation")
    pe.add_argument("--workers", type=int, default=3,
                    help="parallel Extraction-agent calls (default 3; 1 = sequential, gentler on rate limits)")
    pe.set_defaults(fn=cmd_extract)

    sub.add_parser("models").set_defaults(fn=cmd_models)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("serve").set_defaults(fn=cmd_serve)
    sub.add_parser("ui").set_defaults(fn=cmd_ui)
    sub.add_parser("up").set_defaults(fn=cmd_up)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
