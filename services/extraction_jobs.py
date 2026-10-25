"""Bid extraction as a background job (UI "Extract Bid Fields" tab / POST /extract/jobs).

A thin wrapper around the same pipeline as `python main.py extract`: Orchestrator ->
validator -> reconciliation -> report, `write_extraction` and the JSON-lines trace. What it
adds is what an unattended UI run needs:

  preflight      the bid's indexed documents (doc type, addendum number), its output files
                 and whether they exist, the API-key state and the expected request count;
  backups        existing <bid>.* outputs are copied to output/backup_<UTC>_<bid>/ and
                 checked byte for byte BEFORE any model call; a failed backup aborts the run;
  StopGuardLLM   after the first rate-limit / quota, authentication, payment or
                 model-unavailable error no further request is sent in this run, and a
                 cancel request stops the run before the next model call;
  outputs        <bid>.json, .report.md, .trace.jsonl and .cli.log. A run stopped by such an
                 error writes its partial record to output/partial_<UTC>_<bid>/ instead, so
                 a good earlier output is never replaced by an incomplete one.

Bid isolation is the pipeline's own: every search is filtered to the bid id and passages
from other bids are dropped (ExtractionService.gather_evidence).
"""
from __future__ import annotations

import hashlib
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from agents import BidNotIndexedError, Orchestrator
from extraction.schemas import BidExtraction
from extraction.service import write_extraction
from extraction.trace import ExtractionTrace
from jobcontrol import JobCancelled, JobControl, current
from services.llm_service import (LLMConfigError, LLMError, LLMModelUnavailableError, LLMRateLimitError,
                                  ModelNotAllowedError, redact)
from services.uploads import validate_bid_id

ProgressFn = Callable[[int, int, str], None]

OUTPUT_KINDS = {"json": ".json", "report": ".report.md", "trace": ".trace.jsonl", "cli_log": ".cli.log"}
# Observed: Bid2 14 / 24 calls, Bid1 18 calls. Upper bound: every group split twice on truncated
# output (29 calls incl. focused retries and reconciliation) + 3 compact single-field retries.
ESTIMATED_REQUESTS = (14, 32)
# Errors after which another request in the same run would only waste quota or be refused again.
STOP_ERRORS = (LLMRateLimitError, LLMConfigError, ModelNotAllowedError, LLMModelUnavailableError)


class OverwriteNotConfirmed(RuntimeError):
    pass


class BackupError(RuntimeError):
    pass


class ExtractionStopped(RuntimeError):
    """The run stopped on a rate-limit / quota / auth / payment error; partial outputs written aside."""


class RunStopped(LLMError):
    """Raised instead of sending a request after the run was stopped."""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def output_paths(bid_id: str, out_dir: Path) -> dict[str, Path]:
    bid_id = validate_bid_id(bid_id)
    return {kind: Path(out_dir) / f"{bid_id}{suffix}" for kind, suffix in OUTPUT_KINDS.items()}


def indexed_bids(eng) -> set[str]:
    return {b["bid_id"] for b in eng.retrieval.bids() if b.get("documents")}


# ----------------------------------------------------------------------- preflight
def preflight(eng, bid_id: str, out_dir: Optional[Path] = None) -> dict[str, Any]:
    """Everything the UI shows before the user confirms a run. No model call."""
    bid_id = validate_bid_id(bid_id)
    out_dir = Path(out_dir or eng.settings.output_dir)
    docs = [{k: d.get(k) for k in ("file_name", "doc_type", "addendum_number", "document_date", "num_pages",
                                    "chunks", "status", "message")}
            for d in eng.retrieval.documents(bid_id)]
    indexed = [d for d in docs if d["status"] == "indexed"]
    addenda = [d for d in indexed if d["doc_type"] == "addendum"]
    warnings = [f"{d['file_name']}: {d['status']} - {d['message']}" for d in docs
                if d["status"] not in ("indexed", "skipped_duplicate")]
    outputs = {}
    for kind, path in output_paths(bid_id, out_dir).items():
        st = path.stat() if path.exists() else None
        outputs[kind] = {"path": str(path), "exists": st is not None,
                         "modified_utc": _iso(st.st_mtime) if st else None, "size": st.st_size if st else None}
    return {
        "bid_id": bid_id,
        "indexed": bool(indexed),
        "documents": docs,
        "addenda": [{"file_name": d["file_name"], "addendum_number": d["addendum_number"],
                     "document_date": d["document_date"]} for d in addenda],
        "warnings": warnings,
        "outputs": outputs,
        "existing": [o["path"] for o in outputs.values() if o["exists"]],
        "backup_dir_pattern": str(out_dir / f"backup_<UTC time>_{bid_id}"),
        "api_key_configured": bool(eng.settings.openrouter_api_key),
        "default_model": eng.settings.extract_default_model,
        "models": list(eng.settings.llm_models),
        "estimate": {"min_requests": ESTIMATED_REQUESTS[0], "max_requests": ESTIMATED_REQUESTS[1]},
    }


# ----------------------------------------------------------------------- backups
def backup_outputs(paths: dict[str, Path], out_dir: Path, bid_id: str) -> Optional[Path]:
    """Copy the existing output files to a new backup folder and verify each copy byte for byte.
    Never reuses or writes into an existing folder. Raises BackupError (run must not start)."""
    existing = [p for p in paths.values() if p.exists()]
    if not existing:
        return None
    dest = Path(out_dir) / f"backup_{_utc_stamp()}_{bid_id}"
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BackupError(f"backup folder {dest} already exists; not touching it") from exc
    except OSError as exc:
        raise BackupError(f"cannot create backup folder {dest}: {exc}") from exc
    for src in existing:
        target = dest / src.name
        try:
            shutil.copy2(src, target)
        except OSError as exc:
            raise BackupError(f"cannot back up {src.name}: {exc}") from exc
        if _sha256(src) != _sha256(target):
            raise BackupError(f"backup of {src.name} is not identical to the original")
    return dest


# ----------------------------------------------------------------------- LLM guard
class StopGuardLLM:
    """Wraps the engine LLM for one run: checks for cancel before every call and, after the
    first STOP_ERRORS failure, refuses every later call without sending it."""

    def __init__(self, llm, control: Optional[JobControl] = None):
        self.llm = llm
        self.control = control
        self.stopped: Optional[LLMError] = None
        self.sent = 0  # calls handed to the real LLM
        self.refused = 0
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self.control is not None and self.control.cancel_event.is_set()

    def complete(self, messages, model=None):
        with self._lock:
            if self.cancelled:
                self.refused += 1
                raise JobCancelled("Cancelled by user before this model call")
            if self.stopped is not None:
                self.refused += 1
                raise RunStopped(f"not sent: this run stopped after {type(self.stopped).__name__}: {self.stopped}")
            self.sent += 1
        try:
            return self.llm.complete(messages, model=model)
        except STOP_ERRORS as exc:
            with self._lock:
                self.stopped = self.stopped or exc
            raise

    def __getattr__(self, name):  # model_statuses() etc. come from the wrapped LLM
        return getattr(self.llm, name)


# ----------------------------------------------------------------------- progress
class ProgressTrace(ExtractionTrace):
    """The normal trace, plus job progress: finished plan steps / planned steps, and the
    latest model call as the detail line."""

    def __init__(self, bid_id: str, progress: Optional[ProgressFn] = None, control: Optional[JobControl] = None):
        super().__init__(bid_id)
        self._progress, self._control = progress, control
        self._done = self._total = 0
        self._calls = 0

    def event(self, stage: str, **data: Any) -> dict[str, Any]:
        ev = super().event(stage, **data)
        if stage == "plan":
            self._total = len(data.get("steps") or [])
        elif stage == "agent" and data.get("status") in ("done", "failed"):
            self._done += 1
            self._total = max(self._total, self._done + 1)
            self._report(str(data.get("step")))
        elif stage == "done":
            self._total = self._done = max(self._total, self._done)
            self._report("done")
        elif stage == "llm_call":
            self._calls += 1
            if self._control is not None and self._control.on_detail is not None:
                self._control.on_detail(f"model call {self._calls}: {data.get('call')} -> {data.get('outcome')}")
        return ev

    def _report(self, current_step: str) -> None:
        if self._progress is not None:
            self._progress(self._done, self._total, current_step)


# ----------------------------------------------------------------------- .cli.log summary
def summary_lines(record: BidExtraction) -> list[str]:
    """The per-field table and counts line, in the same format `main.py extract` prints."""
    lines = []
    for name, f in record.fields.items():
        value = f.value if not isinstance(f.value, list) else "; ".join(f.value)
        lines.append(f"{f.status:11} {f.confidence:4.2f}  {name:40} {str(value)[:70] if value is not None else '-'}")
    v = record.validation
    lines += ["", f"passed={v.passed} failed={v.failed} (unverified={v.unverified}, error={v.error}) "
                  f"not_found={v.not_found}  addendum_changes={len(record.addendum_changes)}  "
                  f"llm_calls={record.llm_calls}"]
    return lines


# ----------------------------------------------------------------------- run
class ExtractionJobResult(BaseModel):
    bid_id: str
    run_id: str
    model: Optional[str] = None
    started_at: str
    finished_at: str
    outputs: dict[str, str]
    backup_dir: Optional[str] = None
    validation: dict[str, int]
    llm_calls: int = 0
    requests_sent: int = 0
    addendum_changes: list[dict] = Field(default_factory=list)
    fields: list[dict] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        v = self.validation
        return (f"{self.bid_id}: {v.get('passed')} verified, {v.get('failed')} failed, {v.get('not_found')} not found, "
                f"{len(self.addendum_changes)} addendum change(s), {self.llm_calls} LLM calls -> {self.outputs['json']}")


def _write_outputs(state, trace: ExtractionTrace, out_dir: Path, header: list[str]) -> dict[str, str]:
    record = state.final
    paths = output_paths(record.bid_id, out_dir)
    write_extraction(record, out_dir)
    trace.write_jsonl(paths["trace"])
    paths["report"].write_text(redact(state.report_markdown or ""), encoding="utf-8")
    log = header + summary_lines(record) + [f"JSON:   {paths['json']}", f"trace:  {paths['trace']}",
                                            f"report: {paths['report']}"]
    if state.errors:
        log.append(f"agent errors: {state.errors}")
    paths["cli_log"].write_text(redact("\n".join(log) + "\n"), encoding="utf-8")
    return {k: str(p) for k, p in paths.items()}


def run_bid_extraction(eng, bid_id: str, *, model: Optional[str] = None, out_dir: Optional[Path] = None,
                       confirm_overwrite: bool = False, progress: Optional[ProgressFn] = None,
                       workers: int = 1, max_retries: int = 1, **extraction_kwargs: Any) -> ExtractionJobResult:
    """One extraction run with the CLI's documented settings (workers 1, max retries 1,
    reconciliation on). Raises before any model call when the bid is not indexed, outputs
    exist without confirmation, or the backup fails."""
    bid_id = validate_bid_id(bid_id)
    out_dir = Path(out_dir or eng.settings.output_dir)
    model = model or eng.settings.extract_default_model
    known = indexed_bids(eng)
    if bid_id not in known:
        raise BidNotIndexedError(f"bid '{bid_id}' has no indexed documents. Indexed bids: "
                                 f"{', '.join(sorted(known)) or 'none'}")
    paths = output_paths(bid_id, out_dir)
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not confirm_overwrite:
        raise OverwriteNotConfirmed(f"output files already exist ({', '.join(existing)}); confirm overwrite first")
    backup = backup_outputs(paths, out_dir, bid_id)  # before any model call; raises BackupError

    started = datetime.now(timezone.utc)
    control = current()
    guard = StopGuardLLM(eng.llm, control)
    trace = ProgressTrace(bid_id, progress, control)
    orch = Orchestrator(eng.retrieval, guard, qa=eng.qa, indexing=eng.indexing, model=model,
                        max_retries=max_retries, workers=workers, reconcile=True, **extraction_kwargs)
    state = orch.run_extraction(bid_id=bid_id, trace=trace)
    if guard.cancelled:
        raise JobCancelled(f"Cancelled by user after {guard.sent} model request(s); no output files written"
                           + (f" (previous outputs backed up in {backup})" if backup else ""))
    record = state.final
    if record is None:
        raise RuntimeError(f"the run produced no record: {state.errors}")

    docs = [d["file_name"] for d in eng.retrieval.documents(bid_id) if d.get("status") == "indexed"]
    header = [f"Extraction run {state.run_id} for bid '{bid_id}' (UI job)",
              f"model: {record.model or model}  started: {started.isoformat(timespec='seconds')}  "
              f"finished: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
              f"documents ({len(docs)}): {'; '.join(docs)}",
              f"backup of previous outputs: {backup or 'none (no earlier outputs)'}", ""]
    if guard.stopped is not None:
        aside = out_dir / f"partial_{_utc_stamp()}_{bid_id}"
        aside.mkdir(parents=True, exist_ok=False)
        _write_outputs(state, trace, aside, header + [f"STOPPED: {guard.stopped}", ""])
        raise ExtractionStopped(
            f"Stopped after {type(guard.stopped).__name__}: {guard.stopped}. No further model requests were sent "
            f"({guard.sent} sent, {guard.refused} not sent). The incomplete record was written to {aside}; "
            f"{paths['json'].name} was not replaced.")

    outputs = _write_outputs(state, trace, out_dir, header)
    return ExtractionJobResult(
        bid_id=bid_id, run_id=state.run_id, model=record.model or model,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        outputs=outputs, backup_dir=str(backup) if backup else None,
        validation=record.validation.model_dump(), llm_calls=record.llm_calls, requests_sent=guard.sent,
        addendum_changes=[c.model_dump(include={"field", "value", "previous_value", "addendum_number"})
                          for c in record.addendum_changes],
        fields=[{"field": n, "status": f.status, "confidence": f.confidence,
                 "value": f.value if not isinstance(f.value, list) else "; ".join(f.value),
                 "sources": "; ".join(s.citation for s in f.sources), "notes": f.notes}
                for n, f in record.fields.items()],
        documents=docs, errors=list(state.errors),
    )
