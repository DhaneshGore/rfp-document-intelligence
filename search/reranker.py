"""Optional cross-encoder re-ranker (sentence-transformers CrossEncoder)."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol

from config import get_settings
from search.embeddings import _resolve_device

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        s = get_settings()
        self.name = model_name or s.reranker_model
        self.device = _resolve_device(device or s.embedding_device)
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                log.info("Loading reranker %s", self.name)
                self._model = CrossEncoder(self.name, device=self.device, max_length=512)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
