"""Persistent ChromaDB vector store for chunks + metadata."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from search.embeddings import Embedder
from search.filters import to_chroma_where
from schemas import Chunk, SearchFilters

log = logging.getLogger(__name__)


class EmbeddingModelMismatch(RuntimeError):
    pass


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str, embedder: Embedder):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.embedder = embedder
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.name},
            embedding_function=None,
        )
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedder.name and self.collection.count() > 0:
            raise EmbeddingModelMismatch(
                f"Index was built with '{stored_model}' but EMBEDDING_MODEL is '{embedder.name}'. "
                "Re-index with `python main.py index --all --rebuild`."
            )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
            )

    def delete_doc(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": {"$eq": doc_id}})

    def doc_chunk_count(self, doc_id: str) -> int:
        return len(self.collection.get(where={"doc_id": {"$eq": doc_id}}, include=[])["ids"])

    def query(
        self, text: str, n: int, filters: Optional[SearchFilters] = None
    ) -> list[tuple[Chunk, float]]:
        total = self.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(n, total),
            where=to_chroma_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[Chunk, float]] = []
        for cid, doc, md, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append((Chunk.from_store(cid, doc, md), 1.0 - float(dist)))  # cosine similarity
        return out

    def all_chunks(self) -> list[Chunk]:
        res = self.collection.get(include=["documents", "metadatas"])
        return [
            Chunk.from_store(cid, doc, md)
            for cid, doc, md in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def reset(self) -> None:
        name, md = self.collection.name, dict(self.collection.metadata or {})
        md["embedding_model"] = self.embedder.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata=md, embedding_function=None
        )
