"""Background jobs: indexing and bid extraction.

A single worker thread runs jobs one at a time (indexing must stay serialized, and an
extraction must not read the index while it changes), so a
long PDF never blocks API requests: clients submit a job and poll its status, which
includes per-file progress and a page-level detail line. Jobs can be cancelled:
a queued job never starts; a running job stops at the next safe point (between files
or between PDF page batches). Files finished before the cancel stay indexed.
Jobs live in memory; they are lost on restart, but the index itself is persistent and
re-running a job is incremental.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from pydantic import BaseModel

from jobcontrol import JobCancelled, JobControl, set_current

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
FINISHED = ("succeeded", "failed", "cancelled")
JobFn = Callable[[ProgressFn], BaseModel]  # IndexReport for "index" jobs, ExtractionJobResult for "extract"


class JobManager:
    def __init__(self, max_jobs_kept: int = 100):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index-job")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._controls: dict[str, JobControl] = {}
        self._lock = threading.Lock()
        self._max = max_jobs_kept

    def submit(self, description: str, fn: JobFn, kind: str = "index") -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "kind": kind,  # index | extract
            "description": description,
            "status": "queued",  # queued | running | cancelling | succeeded | failed | cancelled
            "progress": {"done": 0, "total": None, "current": None, "detail": None},
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "report": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._controls[job_id] = JobControl(on_detail=lambda d, j=job_id: self._set_detail(j, d))
            self._trim()
        self._pool.submit(self._run, job_id, fn)
        return self.get(job_id)

    def cancel(self, job_id: str) -> Optional[dict[str, Any]]:
        """Request cancellation. Returns the job, or None if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] == "queued":
                job.update(status="cancelled", finished_at=time.time(), error="Cancelled before it started")
            elif job["status"] == "running":
                job["status"] = "cancelling"
                self._controls[job_id].cancel_event.set()
            else:
                return dict(job)
            snapshot = dict(job)  # taken under the lock: the worker may finish the cancel right after
        log.info("job %s cancel requested (%s)", job_id, snapshot["status"])
        return snapshot

    def _set_detail(self, job_id: str, detail: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"]["detail"] = detail
        log.info("job %s: %s", job_id, detail)

    def _run(self, job_id: str, fn: JobFn) -> None:
        def progress(done: int, total: int, current: str) -> None:
            with self._lock:
                self._jobs[job_id]["progress"].update(done=done, total=total, current=current)

        with self._lock:
            if self._jobs[job_id]["status"] == "cancelled":  # cancelled while queued
                return
            self._jobs[job_id].update(status="running", started_at=time.time())
            ctrl = self._controls[job_id]
            desc = self._jobs[job_id]["description"]
        log.info("job %s started: %s", job_id, desc)
        t0 = time.time()
        set_current(ctrl)
        try:
            report = fn(progress)
            with self._lock:
                self._jobs[job_id].update(status="succeeded", report=report.model_dump())
            log.info("job %s succeeded in %.1fs: %s", job_id, time.time() - t0, _describe(report))
        except JobCancelled:
            with self._lock:
                self._jobs[job_id].update(
                    status="cancelled",
                    error="Cancelled by user; files completed before the cancel remain indexed",
                )
            log.info("job %s cancelled after %.1fs", job_id, time.time() - t0)
        except Exception as exc:
            log.exception("Index job %s failed", job_id)
            with self._lock:
                self._jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            set_current(None)
            with self._lock:
                self._jobs[job_id]["finished_at"] = time.time()
                self._controls.pop(job_id, None)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return _copy(job) if job else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((_copy(j) for j in self._jobs.values()), key=lambda j: -j["submitted_at"])

    def queue_position(self, job_id: str) -> int:
        """Number of unfinished jobs submitted before this one."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return 0
            return sum(1 for j in self._jobs.values()
                       if j["status"] not in FINISHED and j["submitted_at"] < job["submitted_at"])

    def _trim(self) -> None:
        finished = sorted(
            (j for j in self._jobs.values() if j["status"] in FINISHED),
            key=lambda j: j["submitted_at"],
        )
        while len(self._jobs) > self._max and finished:
            self._jobs.pop(finished.pop(0)["job_id"], None)


def _describe(report: BaseModel) -> str:
    if hasattr(report, "total_chunks_added"):  # IndexReport
        return f"{report.summary or 'no files'}, {report.total_chunks_added} chunks added"
    return str(getattr(report, "summary", "") or "done")


def _copy(job: dict[str, Any]) -> dict[str, Any]:
    return {**job, "progress": dict(job["progress"])}
