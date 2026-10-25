"""Embedding models behind a small protocol so tests can inject a fake."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol

from config import get_settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (default: BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        query_instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        s = get_settings()
        self.name = model_name or s.embedding_model
        self.device = _resolve_device(device or s.embedding_device)
        self.query_instruction = (
            s.embedding_query_instruction if query_instruction is None else query_instruction
        )
        self.batch_size = batch_size or s.embedding_batch_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                log.info("Loading embedding model %s on %s", self.name, self.device)
                self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [f"{self.query_instruction}{text}"], normalize_embeddings=True, show_progress_bar=False
        )
        return vec[0].tolist()
