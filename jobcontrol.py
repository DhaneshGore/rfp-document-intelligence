"""Cooperative cancellation and progress detail for background jobs.

The job worker installs a JobControl for its thread. Long-running code (the indexing
loop, Docling page batches) calls `check_cancelled()` at safe points and may call
`report_detail()` with fine-grained progress. Outside a job both are no-ops, so the
CLI and tests are unaffected.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional


class JobCancelled(Exception):
    """Raised at a safe point after a user asked to cancel the current job."""


class JobControl:
    def __init__(self, on_detail: Optional[Callable[[str], None]] = None):
        self.cancel_event = threading.Event()
        self.on_detail = on_detail


_local = threading.local()


def set_current(ctrl: Optional[JobControl]) -> None:
    _local.ctrl = ctrl


def current() -> Optional[JobControl]:
    return getattr(_local, "ctrl", None)


def check_cancelled() -> None:
    ctrl = current()
    if ctrl is not None and ctrl.cancel_event.is_set():
        raise JobCancelled("Cancelled by user")


def report_detail(text: str) -> None:
    ctrl = current()
    if ctrl is not None and ctrl.on_detail is not None:
        ctrl.on_detail(text)
