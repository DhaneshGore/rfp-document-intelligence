"""Metadata filter helpers shared by the vector store and BM25 index."""
from __future__ import annotations

from typing import Any, Optional

from schemas import Chunk, SearchFilters


def to_chroma_where(filters: Optional[SearchFilters]) -> Optional[dict[str, Any]]:
    if filters is None:
        return None
    active = filters.active()
    if not active:
        return None
    clauses = [{k: {"$eq": v}} for k, v in active.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches(chunk: Chunk, filters: Optional[SearchFilters]) -> bool:
    if filters is None:
        return True
    for key, val in filters.active().items():
        attr = getattr(chunk, key, None)
        attr = getattr(attr, "value", attr)  # enums
        if attr != val:
            return False
    return True
