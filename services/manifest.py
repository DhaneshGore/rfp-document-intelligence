"""Persistent record of indexed files (path -> content hash, doc id, status...).

Used for incremental indexing and deduplication. Every mutation bumps `version`,
which retrieval uses to know when its in-memory BM25 index is stale (also across
processes, since the file is re-read when its mtime changes).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_path(p: str | Path) -> str:
    return os.path.normcase(str(Path(p).resolve()))


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": 0, "updated_at": None, "files": {}}
        self._mtime: Optional[int] = None
        self.reload(force=True)

    # ------------------------------------------------------------------ io
    def reload(self, force: bool = False) -> bool:
        """Re-read from disk if it changed. Returns True if reloaded."""
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime_ns
            except FileNotFoundError:
                return False
            if not force and mtime == self._mtime:
                return False
            with open(self.path, encoding="utf-8") as fh:
                self._data = json.load(fh)
            self._mtime = mtime
            return True

    def _save(self) -> None:
        self._data["version"] = int(self._data.get("version", 0)) + 1
        self._data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp, self.path)  # atomic on the same volume
        self._mtime = self.path.stat().st_mtime_ns

    # ------------------------------------------------------------------ api
    @property
    def version(self) -> int:
        return int(self._data.get("version", 0))

    def files(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._data.get("files", {}))

    def get(self, source_path: str | Path) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._data["files"].get(norm_path(source_path))

    def find_by_hash(self, bid_id: str, content_hash: str, exclude: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            for key, rec in self._data["files"].items():
                if (
                    key != exclude
                    and rec.get("bid_id") == bid_id
                    and rec.get("content_hash") == content_hash
                    and rec.get("status") == "indexed"
                ):
                    return rec
        return None

    def upsert(self, source_path: str | Path, record: dict[str, Any]) -> None:
        with self._lock:
            record = {**record, "source_path": str(Path(source_path).resolve()), "updated_at": _now()}
            self._data["files"][norm_path(source_path)] = record
            self._save()

    def remove(self, source_path: str | Path) -> Optional[dict[str, Any]]:
        with self._lock:
            rec = self._data["files"].pop(norm_path(source_path), None)
            if rec is not None:
                self._save()
            return rec

    def clear(self) -> None:
        with self._lock:
            self._data["files"] = {}
            self._save()

    def bids(self) -> dict[str, dict[str, Any]]:
        """Per-bid summary of indexed documents."""
        out: dict[str, dict[str, Any]] = {}
        for rec in self.files().values():
            b = out.setdefault(rec["bid_id"], {"bid_id": rec["bid_id"], "documents": 0, "chunks": 0, "failed": 0})
            if rec.get("status") == "indexed":
                b["documents"] += 1
                b["chunks"] += int(rec.get("chunks", 0))
            elif rec.get("status") == "failed":
                b["failed"] += 1
        return dict(sorted(out.items()))
