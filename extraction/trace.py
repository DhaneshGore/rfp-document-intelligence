"""Structured extraction trace: one JSON event per stage (JSON-lines file).

Stages: plan, retrieve, llm_call, validate, retry, reconcile, done. Events hold evidence
references (chunk id, file, page, doc type), validation outcomes, retries, token/latency
figures and errors - never prompts, passage text or configuration. Every serialized line
goes through `redact`, so a key echoed in an error message cannot reach the file.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from schemas import SearchResult
from services.llm_service import redact


def evidence_refs(evidence: list[SearchResult]) -> list[dict[str, Any]]:
    return [{"n": i, "chunk_id": r.chunk_id, "file": r.file_name, "page": r.page_number,
             "doc_type": r.doc_type, "addendum_number": r.addendum_number}
            for i, r in enumerate(evidence, start=1)]


class ExtractionTrace:
    def __init__(self, bid_id: str):
        self.bid_id = bid_id
        self.events: list[dict[str, Any]] = []
        self._t0 = time.time()
        self._lock = threading.Lock()  # agents may run in parallel threads

    def event(self, stage: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            ev = {"seq": len(self.events) + 1, "t": round(time.time() - self._t0, 3), "bid_id": self.bid_id,
                  "stage": stage, **data}
            self.events.append(ev)
        return ev

    def stages(self) -> list[str]:
        return [e["stage"] for e in self.events]

    def to_jsonl(self) -> str:
        return "".join(redact(json.dumps(e, ensure_ascii=False, default=str)) + "\n" for e in self.events)

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl(), encoding="utf-8")
        return path
