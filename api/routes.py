"""FastAPI application exposing indexing, search and grounded Q&A.

Run with:  python main.py serve   (or: uvicorn api.routes:app)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import config  # noqa: F401  (sets USE_TF before heavy imports)
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from schemas import Answer, IndexReport, SearchFilters, SearchResult
from search.hybrid_search import SEARCH_MODES
from services.container import Engine, get_engine
from services.extraction_jobs import preflight, run_bid_extraction
from services.jobs import JobManager
from services.llm_service import OpenRouterLLM
from services.uploads import UploadError, inspect_pdf, store_upload, validate_bid_id

config.configure_logging()
log = logging.getLogger(__name__)

app = FastAPI(
    title="RFP Intelligence Platform - Search API",
    version="1.0.0",
    description="Hybrid (dense + BM25 + RRF + cross-encoder) search over bid documents with cited answers, "
                "and background extraction of the 20 bid fields to JSON.",
)


# ----------------------------------------------------------------------- request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request with status and duration. Successful GETs other than /search
    (the UI polls /health, /bids, /documents and job status constantly) are DEBUG."""
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.error("%s %s -> 500 (%.0f ms)", request.method, request.url.path, (time.perf_counter() - t0) * 1000)
        raise  # uvicorn logs the traceback
    ms = (time.perf_counter() - t0) * 1000
    code = response.status_code
    if code >= 500:
        level = logging.ERROR
    elif code >= 400:
        level = logging.WARNING
    elif request.method == "GET" and request.url.path != "/search":
        level = logging.DEBUG
    else:
        level = logging.INFO
    log.log(level, "%s %s -> %d (%.0f ms)", request.method, request.url.path, code, ms)
    return response


@app.exception_handler(StarletteHTTPException)
async def _log_http_error(request: Request, exc: StarletteHTTPException):
    log.warning("%s %s rejected (%d): %s", request.method, request.url.path, exc.status_code, exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _log_validation_error(request: Request, exc: RequestValidationError):
    problems = "; ".join(f"{'.'.join(map(str, e.get('loc', ())))}: {e.get('msg')}" for e in exc.errors())
    log.warning("%s %s invalid request: %s", request.method, request.url.path, problems)
    return await request_validation_exception_handler(request, exc)


def engine() -> Engine:
    return get_engine()


_jobs = JobManager()


def jobs() -> JobManager:
    return _jobs


# ----------------------------------------------------------------------- models
class IndexRequest(BaseModel):
    all: bool = Field(False, description="Index every discovered bid folder and upload folder")
    bid_id: Optional[str] = Field(None, description="Index all folders of this bid (or bid id for `paths`)")
    bid_path: Optional[str] = Field(None, description="Path to a bid folder (inside BIDS_ROOT or data/)")
    paths: Optional[list[str]] = Field(None, description="Specific files (inside BIDS_ROOT or data/)")
    force: bool = False
    rebuild: bool = False
    ocr: Optional[bool] = Field(
        None, description="Override PDF OCR for this request (false = faster, scanned pages stay empty)"
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    bid_id: Optional[str] = None
    doc_type: Optional[str] = None
    addendum_number: Optional[int] = None
    file_name: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Evidence passages given to the LLM")
    model: Optional[str] = None
    mode: Optional[str] = None


class ExtractRequest(BaseModel):
    bid_id: str
    model: Optional[str] = Field(None, description="Configured free model (default: EXTRACT_DEFAULT_MODEL)")
    confirm_overwrite: bool = Field(False, description="Required when output files for the bid already exist; "
                                                       "they are backed up first")


class SearchResponse(BaseModel):
    query: str
    mode: str
    filters: dict
    results: list[SearchResult]


class UploadResponse(BaseModel):
    bid_id: str
    saved: list[dict]
    rejected: list[dict]
    report: Optional[IndexReport] = None


def _allowed_path(p: str, eng: Engine) -> Path:
    """Only allow paths inside the bids root or the data directory (no traversal)."""
    path = Path(p)
    if not path.is_absolute():
        path = eng.settings.bids_root / path
    path = path.resolve()
    roots = [eng.settings.bids_root.resolve(), eng.settings.data_dir.resolve()]
    if not any(path == r or r in path.parents for r in roots):
        raise HTTPException(400, f"Path outside allowed roots: {p}")
    if not path.exists():
        raise HTTPException(404, f"Path not found: {p}")
    return path


# ----------------------------------------------------------------------- routes
@app.get("/health")
def health(eng: Engine = Depends(engine)) -> dict:
    return {"status": "ok", **eng.retrieval.stats()}


@app.get("/bids")
def bids(eng: Engine = Depends(engine)) -> dict:
    indexed = {b["bid_id"]: b for b in eng.retrieval.bids()}
    discovered = sorted(eng.indexing.discover_bid_folders())
    for b in discovered:
        indexed.setdefault(b, {"bid_id": b, "documents": 0, "chunks": 0, "failed": 0})
    return {"bids": sorted(indexed.values(), key=lambda b: b["bid_id"])}


@app.get("/documents")
def documents(bid_id: Optional[str] = None, eng: Engine = Depends(engine)) -> dict:
    return {"documents": eng.retrieval.documents(bid_id)}


@app.get("/models")
def models(eng: Engine = Depends(engine)) -> dict:
    return {
        "default": eng.settings.llm_default_model,
        "fallbacks": eng.settings.llm_fallback_models,
        "api_key_configured": bool(eng.settings.openrouter_api_key),
        "models": eng.llm.model_statuses(),
    }


def _plan_index(req: IndexRequest, eng: Engine) -> tuple[str, Callable[[Callable], IndexReport]]:
    """Validate an index request up front; return (description, fn(progress) -> report)."""
    ix = eng.indexing
    ocr = req.ocr
    note = "" if ocr is None else (" (OCR on)" if ocr else " (OCR off)")
    try:
        if req.rebuild:
            return "rebuild", lambda p: ix.rebuild(progress=p)
        if req.paths:
            if not req.bid_id:
                raise HTTPException(400, "bid_id is required with paths")
            bid = validate_bid_id(req.bid_id)
            files = [_allowed_path(x, eng) for x in req.paths]
            return (f"{len(files)} file(s) -> {bid}{note}",
                    lambda p: ix.index_files(files, bid, req.force, progress=p, ocr=ocr))
        if req.bid_path:
            folder = _allowed_path(req.bid_path, eng)
            if not folder.is_dir():
                raise HTTPException(400, f"Not a folder: {req.bid_path}")
            bid = validate_bid_id(req.bid_id or folder.name)
            return (f"folder {folder.name}{note}",
                    lambda p: ix.index_folder(folder, bid, req.force, progress=p, ocr=ocr))
        if req.bid_id:
            bid = validate_bid_id(req.bid_id)
            if not ix.folders_for_bid(bid):
                raise HTTPException(404, f"No folder found for bid '{bid}'")
            return f"bid {bid}{note}", lambda p: ix.index_bid(bid, req.force, progress=p, ocr=ocr)
        if req.all:
            return f"all bids{note}", lambda p: ix.index_all(req.force, progress=p, ocr=ocr)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    raise HTTPException(400, "Provide one of: all, bid_id, bid_path, paths, rebuild")


@app.post("/index", response_model=IndexReport)
async def index(req: IndexRequest, eng: Engine = Depends(engine)) -> IndexReport:
    """Synchronous indexing (waits for completion). Prefer /index/jobs for large PDFs."""
    _, fn = _plan_index(req, eng)
    try:
        return await run_in_threadpool(fn, None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/index/jobs", status_code=202)
def submit_index_job(req: IndexRequest, eng: Engine = Depends(engine)) -> dict:
    """Start indexing in the background; poll GET /index/jobs/{job_id}."""
    desc, fn = _plan_index(req, eng)
    return jobs().submit(desc, fn)


@app.get("/index/jobs")
def list_index_jobs() -> dict:
    return {"jobs": jobs().list()}


@app.get("/index/jobs/{job_id}")
def get_index_job(job_id: str) -> dict:
    job = jobs().get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job '{job_id}'")
    job["queue_position"] = jobs().queue_position(job_id)
    return job


@app.post("/index/jobs/{job_id}/cancel")
def cancel_index_job(job_id: str) -> dict:
    """Cancel a job: a queued job never starts; a running job stops at the next safe
    point (between files or PDF page batches). Finished files stay indexed."""
    job = jobs().cancel(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job '{job_id}'")
    return job


@app.get("/extract/preflight")
def extract_preflight(bid_id: str, eng: Engine = Depends(engine)) -> dict:
    """What an extraction run for `bid_id` would use and write. No model call."""
    try:
        return preflight(eng, bid_id)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/extract/jobs", status_code=202)
def submit_extract_job(req: ExtractRequest, eng: Engine = Depends(engine)) -> dict:
    """Start extraction for one indexed bid in the background (same queue as indexing);
    poll GET /index/jobs/{job_id}. Never started implicitly by upload or indexing."""
    try:
        pre = preflight(eng, req.bid_id)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not pre["indexed"]:
        raise HTTPException(404, f"Bid '{pre['bid_id']}' has no indexed documents")
    if isinstance(eng.llm, OpenRouterLLM) and not pre["api_key_configured"]:
        raise HTTPException(400, "OPENROUTER_API_KEY is not set; extraction needs the LLM")
    model = req.model or pre["default_model"]
    if model not in pre["models"]:
        raise HTTPException(400, f"Model '{model}' is not one of the configured free models")
    if pre["existing"] and not req.confirm_overwrite:
        raise HTTPException(409, f"Output files already exist ({', '.join(Path(p).name for p in pre['existing'])}); "
                                 "confirm overwrite (they are backed up first)")
    bid, out = pre["bid_id"], pre["outputs"]["json"]["path"]
    return jobs().submit(
        f"extract {bid} -> {out} ({model})",
        lambda p: run_bid_extraction(eng, bid, model=model, confirm_overwrite=req.confirm_overwrite, progress=p),
        kind="extract")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    bid_id: str = Form(...),
    files: list[UploadFile] = File(...),
    index: bool = Form(True),
    ocr: Optional[bool] = Form(None),
    eng: Engine = Depends(engine),
) -> UploadResponse:
    """Store files. Each saved PDF includes a profile (pages, images, pages without text,
    estimated seconds with OCR, `large` flag) so clients can warn before indexing."""
    try:
        bid_id = validate_bid_id(bid_id)
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    max_bytes = eng.settings.max_upload_mb * 1024 * 1024
    saved, rejected, paths = [], [], []
    for f in files:
        data = await f.read(max_bytes + 1)
        try:
            path, existed = store_upload(bid_id, f.filename or "", data)
            entry = {"original_name": f.filename, "stored_as": path.name,
                     "path": str(path), "already_existed": existed}
            if path.suffix.lower() == ".pdf":
                try:
                    entry["pdf"] = inspect_pdf(path)
                except Exception as exc:  # profile is advisory; indexing reports real errors
                    entry["pdf"] = {"error": f"{type(exc).__name__}: {exc}"}
            saved.append(entry)
            paths.append(path)
        except UploadError as exc:
            rejected.append({"original_name": f.filename, "error": str(exc)})
    report = None
    if index and paths:
        report = await run_in_threadpool(lambda: eng.indexing.index_files(paths, bid_id, ocr=ocr))
    return UploadResponse(bid_id=bid_id, saved=saved, rejected=rejected, report=report)


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=1000),
    bid_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    addendum_number: Optional[int] = None,
    file_name: Optional[str] = None,
    top_k: int = Query(5, ge=1, le=50),
    mode: Optional[str] = Query(None, description=f"one of {SEARCH_MODES}"),
    expand: Optional[bool] = None,
    eng: Engine = Depends(engine),
) -> SearchResponse:
    filters = SearchFilters(bid_id=bid_id, doc_type=doc_type, addendum_number=addendum_number, file_name=file_name)
    mode = mode or eng.settings.default_search_mode
    t0 = time.perf_counter()
    try:
        results = await run_in_threadpool(eng.retrieval.search, q, top_k, filters, mode, expand)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    log.info(
        'search mode=%s top_k=%d filters=%s results=%d top=%s %.0f ms q="%s"',
        mode, top_k, filters.active() or "-", len(results),
        f"[{results[0].citation}] {results[0].score:.3f}" if results else "-",
        (time.perf_counter() - t0) * 1000, q[:100],
    )
    return SearchResponse(query=q, mode=mode, filters=filters.active(), results=results)


@app.post("/ask", response_model=Answer)
async def ask(req: AskRequest, eng: Engine = Depends(engine)) -> Answer:
    filters = SearchFilters(
        bid_id=req.bid_id, doc_type=req.doc_type, addendum_number=req.addendum_number, file_name=req.file_name
    )
    t0 = time.perf_counter()
    try:
        ans = await run_in_threadpool(eng.qa.ask, req.question, filters, req.top_k, req.model, req.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    u = ans.usage
    log.info(
        'ask status=%s finish=%s model=%s citations=%d evidence=%d llm=%ss tokens=%s/%s cost=%s total=%.1fs q="%s"',
        ans.status, ans.finish_reason or "-", ans.model or "-", len(ans.citations), len(ans.evidence), ans.latency_s if ans.latency_s is not None else "-",
        u.prompt_tokens if u else "-", u.completion_tokens if u else "-", u.cost if u else "-",
        time.perf_counter() - t0, req.question[:100],
    )
    if ans.error:  # LLM errors arrive here already redacted (no API key)
        log.warning("ask problem: %s", ans.error)
    return ans
