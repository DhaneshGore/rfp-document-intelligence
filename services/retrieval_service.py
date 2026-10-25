"""Retrieval facade: owns the BM25 cache and exposes search + index metadata.

This is the "search tool" that the Phase-2 agents will call.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from search.hybrid_search import HybridSearcher
from search.keyword_search import BM25Index
from search.reranker import Reranker
from search.vector_store import ChromaVectorStore
from schemas import SearchFilters, SearchResult
from services.manifest import Manifest

log = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        manifest: Manifest,
        reranker: Optional[Reranker] = None,
    ):
        self.vector_store = vector_store
        self.manifest = manifest
        self._bm25: Optional[BM25Index] = None
        self._bm25_version: Optional[int] = None
        self._lock = threading.Lock()
        self.searcher = HybridSearcher(vector_store, self._get_bm25, reranker)

    def _get_bm25(self) -> BM25Index:
        """Rebuild BM25 from the vector store whenever the manifest version changes."""
        with self._lock:
            self.manifest.reload()
            if self._bm25 is None or self._bm25_version != self.manifest.version:
                chunks = self.vector_store.all_chunks()
                self._bm25 = BM25Index(chunks)
                self._bm25_version = self.manifest.version
                log.info("BM25 index rebuilt: %d chunks (manifest v%s)", len(chunks), self._bm25_version)
            return self._bm25

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        mode: Optional[str] = None,
        expand: Optional[bool] = None,
    ) -> list[SearchResult]:
        return self.searcher.search(query, top_k=top_k, filters=filters, mode=mode, expand=expand)

    def bids(self) -> list[dict]:
        self.manifest.reload()
        return list(self.manifest.bids().values())

    def documents(self, bid_id: Optional[str] = None) -> list[dict]:
        self.manifest.reload()
        docs = [r for r in self.manifest.files().values() if bid_id in (None, r.get("bid_id"))]
        return sorted(docs, key=lambda r: (r.get("bid_id", ""), r.get("file_name", "")))

    def stats(self) -> dict:
        self.manifest.reload()
        return {
            "chunks": self.vector_store.count(),
            "documents": sum(1 for r in self.manifest.files().values() if r.get("status") == "indexed"),
            "bids": len(self.manifest.bids()),
            "manifest_version": self.manifest.version,
        }
