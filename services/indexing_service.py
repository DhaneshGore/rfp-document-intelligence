"""Incremental indexing of bid folders and uploaded files.

* A file is re-processed only when its content hash changed (or `force`).
* The same bytes uploaded twice into one bid are detected and skipped.
* Chunk ids are deterministic (bid + content hash), so re-runs upsert instead of
  duplicating.
* Files that disappeared from an indexed folder are pruned from the index.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from config import get_settings
from ingestion.chunker import Chunker
from ingestion.loader import DocumentLoader
from ingestion.metadata import file_sha256, make_doc_id
from jobcontrol import JobCancelled, check_cancelled
from search.vector_store import ChromaVectorStore
from schemas import FileIndexStatus, IndexReport
from services.manifest import Manifest, norm_path
from services.uploads import validate_bid_id

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


class IndexingService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        vector_store: ChromaVectorStore,
        manifest: Manifest,
    ):
        self.loader = loader
        self.chunker = chunker
        self.vector_store = vector_store
        self.manifest = manifest
        self.settings = get_settings()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ discovery
    def supported_files(self, folder: Path) -> list[Path]:
        exts = set(self.settings.allowed_extensions)
        return sorted(
            p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts
        )

    def discover_bid_folders(self) -> dict[str, list[Path]]:
        """bid_id -> folders (original bid folder and/or uploads/<bid_id>)."""
        s = self.settings
        found: dict[str, list[Path]] = {}
        for d in sorted(s.bids_root.glob(s.bid_folder_glob)):
            if d.is_dir():
                found.setdefault(d.name, []).append(d)
        if s.uploads_dir.exists():
            for d in sorted(s.uploads_dir.iterdir()):
                if d.is_dir():
                    found.setdefault(d.name, []).append(d)
        return found

    def folders_for_bid(self, bid_id: str) -> list[Path]:
        return self.discover_bid_folders().get(bid_id, [])

    # ------------------------------------------------------------------ indexing
    def index_all(
        self, force: bool = False, progress: Optional[ProgressFn] = None, ocr: Optional[bool] = None
    ) -> IndexReport:
        report = IndexReport()
        t0 = time.time()
        for bid_id, folders in self.discover_bid_folders().items():
            for folder in folders:
                sub = self.index_folder(folder, bid_id=bid_id, force=force, progress=progress, ocr=ocr)
                report.files += sub.files
                report.total_chunks_added += sub.total_chunks_added
        report.seconds = round(time.time() - t0, 2)
        return report

    def index_bid(
        self, bid_id: str, force: bool = False, progress: Optional[ProgressFn] = None, ocr: Optional[bool] = None
    ) -> IndexReport:
        folders = self.folders_for_bid(bid_id)
        if not folders:
            raise FileNotFoundError(f"No folder found for bid '{bid_id}'")
        report = IndexReport()
        for folder in folders:
            sub = self.index_folder(folder, bid_id=bid_id, force=force, progress=progress, ocr=ocr)
            report.files += sub.files
            report.total_chunks_added += sub.total_chunks_added
        return report

    def index_folder(
        self,
        folder: Path,
        bid_id: Optional[str] = None,
        force: bool = False,
        prune: bool = True,
        progress: Optional[ProgressFn] = None,
        ocr: Optional[bool] = None,
    ) -> IndexReport:
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Bid folder not found: {folder}")
        bid_id = validate_bid_id(bid_id or folder.name)
        files = self.supported_files(folder)
        report = self.index_files(files, bid_id, force=force, progress=progress, ocr=ocr)
        if prune:
            report.files += self._prune_missing(folder)
        return report

    def index_files(
        self,
        paths: Iterable[Path],
        bid_id: str,
        force: bool = False,
        progress: Optional[ProgressFn] = None,
        ocr: Optional[bool] = None,
    ) -> IndexReport:
        """Index files one by one. Inside a background job, a cancel request stops the
        loop between files (and between PDF page batches); finished files stay indexed."""
        bid_id = validate_bid_id(bid_id)
        paths = [Path(p) for p in paths]
        report = IndexReport()
        t0 = time.time()
        for i, p in enumerate(paths, start=1):
            check_cancelled()
            if progress:
                progress(i - 1, len(paths), p.name)
            status = self.index_file(p, bid_id, force=force, ocr=ocr)
            report.files.append(status)
            if status.status == "indexed":
                report.total_chunks_added += status.chunks
        if progress:
            progress(len(paths), len(paths), "done")
        report.seconds = round(time.time() - t0, 2)
        return report

    def index_file(
        self, path: Path, bid_id: str, force: bool = False, ocr: Optional[bool] = None
    ) -> FileIndexStatus:
        path = Path(path).resolve()
        t0 = time.time()
        status = FileIndexStatus(source_path=str(path), bid_id=bid_id, file_name=path.name, status="failed")
        with self._lock:
            try:
                if path.suffix.lower() not in self.settings.allowed_extensions:
                    raise ValueError(f"Unsupported file type: {path.suffix}")
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")
                content_hash = file_sha256(path)
                doc_id = make_doc_id(bid_id, content_hash)
                self.manifest.reload()
                prev = self.manifest.get(path)

                if (
                    prev
                    and not force
                    and prev.get("status") == "indexed"
                    and prev.get("content_hash") == content_hash
                    and prev.get("bid_id") == bid_id
                    and self.vector_store.doc_chunk_count(doc_id) > 0
                ):
                    status.status, status.chunks = "skipped_unchanged", int(prev.get("chunks", 0))
                    status.doc_type = prev.get("doc_type")
                    status.message = "Unchanged since last index"
                    return status

                dup = self.manifest.find_by_hash(bid_id, content_hash, exclude=norm_path(path))
                if dup and not force:
                    status.status = "skipped_duplicate"
                    status.doc_type = dup.get("doc_type")
                    status.message = f"Identical content already indexed as '{dup.get('file_name')}'"
                    self.manifest.upsert(path, self._record(bid_id, path, content_hash, doc_id, status,
                                                            duplicate_of=dup.get("source_path")))
                    return status

                doc = self.loader.load(path, bid_id, content_hash, ocr=ocr)
                status.doc_type = doc.metadata.doc_type.value
                status.warnings = list(doc.warnings)
                if doc.is_empty:
                    hint = next((w for w in doc.warnings if "JavaScript" in w), "")
                    raise ValueError("No extractable text in document" + (f". {hint}" if hint else ""))
                chunks = self.chunker.chunk(doc)
                if not chunks:
                    raise ValueError("Document produced no chunks")
                check_cancelled()  # last safe point before the index is modified
                if prev and prev.get("doc_id") and prev.get("doc_id") != doc_id:
                    # Content changed: replace the old version's chunks (only after the new
                    # version parsed successfully, so a failure/cancel keeps the old one).
                    self.vector_store.delete_doc(prev["doc_id"])
                self.vector_store.delete_doc(doc_id)  # clear partial leftovers
                self.vector_store.add_chunks(chunks)
                status.status, status.chunks = "indexed", len(chunks)
                status.message = f"Indexed {len(chunks)} chunks ({doc.metadata.parser})"
                self.manifest.upsert(
                    path,
                    self._record(
                        bid_id, path, content_hash, doc_id, status,
                        doc_type=doc.metadata.doc_type.value,
                        addendum_number=doc.metadata.addendum_number,
                        document_date=doc.metadata.document_date,
                        num_pages=doc.metadata.num_pages,
                        parser=doc.metadata.parser,
                        ocr_used=doc.metadata.ocr_used,
                        empty_pages=doc.empty_pages,
                    ),
                )
            except JobCancelled:
                status.status = "cancelled"  # for the log line only; not recorded
                raise  # not a failure of this file: leave the manifest untouched
            except Exception as exc:
                log.exception("Indexing failed for %s", path)
                status.status = "failed"
                status.message = f"{type(exc).__name__}: {exc}"
                try:
                    self.manifest.upsert(
                        path,
                        self._record(bid_id, path, "", "", status),
                    )
                except Exception:  # pragma: no cover - manifest write failure
                    log.exception("Could not record failure in manifest")
            finally:
                status.seconds = round(time.time() - t0, 2)
                log.info("index %-17s %s/%s  chunks=%d  %.1fs  %s", status.status, bid_id, path.name,
                         status.chunks, status.seconds, status.message)
                for w in status.warnings:
                    log.info("  warning (%s): %s", path.name, w)
        return status

    # ------------------------------------------------------------------ maintenance
    def _prune_missing(self, folder: Path) -> list[FileIndexStatus]:
        removed: list[FileIndexStatus] = []
        prefix = norm_path(folder)
        for key, rec in self.manifest.files().items():
            if key.startswith(prefix) and Path(rec["source_path"]).parent.resolve() == folder:
                if not Path(rec["source_path"]).exists():
                    removed.append(self.remove_document(rec["source_path"]))
        return removed

    def remove_document(self, source_path: str | Path) -> FileIndexStatus:
        with self._lock:
            rec = self.manifest.remove(source_path)
            if rec and rec.get("doc_id"):
                # Only delete chunks if no other manifest entry still points at this doc.
                still_used = any(r.get("doc_id") == rec["doc_id"] and r.get("status") == "indexed"
                                 for r in self.manifest.files().values())
                if not still_used:
                    self.vector_store.delete_doc(rec["doc_id"])
        return FileIndexStatus(
            source_path=str(source_path),
            bid_id=(rec or {}).get("bid_id", ""),
            file_name=Path(source_path).name,
            status="removed",
            message="File no longer exists; removed from index" if rec else "Not in index",
        )

    def rebuild(self, progress: Optional[ProgressFn] = None) -> IndexReport:
        with self._lock:
            self.vector_store.reset()
            self.manifest.clear()
        return self.index_all(force=True, progress=progress)

    @staticmethod
    def _record(bid_id, path: Path, content_hash, doc_id, status: FileIndexStatus, **extra) -> dict:
        return {
            "bid_id": bid_id,
            "file_name": path.name,
            "content_hash": content_hash,
            "doc_id": doc_id,
            "size": path.stat().st_size if path.exists() else None,
            "status": status.status,
            "chunks": status.chunks,
            "message": status.message,
            "warnings": status.warnings,
            "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }
