"""Wires the components together (one shared instance per process)."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config import Settings, configure_logging, get_settings
from ingestion.chunker import Chunker
from ingestion.loader import DocumentLoader
from search.embeddings import Embedder, SentenceTransformerEmbedder
from search.reranker import CrossEncoderReranker, Reranker
from search.vector_store import ChromaVectorStore
from services.indexing_service import IndexingService
from services.llm_service import OpenRouterLLM
from services.manifest import Manifest
from services.qa_service import QAService
from services.retrieval_service import RetrievalService


@dataclass
class Engine:
    settings: Settings
    vector_store: ChromaVectorStore
    manifest: Manifest
    indexing: IndexingService
    retrieval: RetrievalService
    llm: OpenRouterLLM
    qa: QAService


def build_engine(
    settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None,
    reranker: Optional[Reranker] = None,
    loader: Optional[DocumentLoader] = None,
    llm: Optional[OpenRouterLLM] = None,
) -> Engine:
    s = settings or get_settings()
    s.ensure_dirs()
    embedder = embedder or SentenceTransformerEmbedder()
    if reranker is None and s.reranker_enabled:
        reranker = CrossEncoderReranker()
    vector_store = ChromaVectorStore(s.chroma_dir, s.collection_name, embedder)
    manifest = Manifest(s.manifest_path)
    indexing = IndexingService(loader or DocumentLoader(), Chunker(), vector_store, manifest)
    retrieval = RetrievalService(vector_store, manifest, reranker)
    llm = llm or OpenRouterLLM(s)
    return Engine(s, vector_store, manifest, indexing, retrieval, llm, QAService(retrieval, llm))


log = logging.getLogger(__name__)
_engine: Optional[Engine] = None
_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    with _lock:
        if _engine is None:
            configure_logging()
            t0 = time.time()
            _engine = build_engine()
            st = _engine.retrieval.stats()
            log.info(
                "Engine ready in %.1fs: %d chunks, %d documents, %d bids; default model %s; OpenRouter key %s",
                time.time() - t0, st["chunks"], st["documents"], st["bids"], _engine.settings.llm_default_model,
                "configured" if _engine.settings.openrouter_api_key else "NOT configured",
            )
        return _engine
