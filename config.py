"""Central configuration, loaded from environment variables / `.env`.

Every tunable (paths, models, chunking, retrieval and LLM settings) lives here so
that no module hard-codes bid names, file names or model IDs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# The global Python on some machines ships an old TensorFlow that is incompatible
# with modern protobuf; `transformers` imports it opportunistically and crashes.
# This project only uses PyTorch, so disable the TF backend before anything imports
# transformers (sentence-transformers, Docling).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # ChromaDB telemetry off

import json  # noqa: E402
from typing import Annotated  # noqa: E402

from pydantic import Field, field_validator  # noqa: E402
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict  # noqa: E402

# Lists may be given in env/.env as JSON (["a","b"]) or comma-separated (a,b).
StrList = Annotated[list[str], NoDecode]

PROJECT_ROOT = Path(__file__).resolve().parent

# Default free model set. Each ID was checked against https://openrouter.ai/api/v1/models
# (prompt and completion price both "0") on 2026-10-23. Free status is re-verified at
# runtime by services.llm_service before any call, because OpenRouter changes it often.
DEFAULT_FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "thinkingmachines/inkling:free",
    "nvidia/nemotron-3.5-lightning:free",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Paths -------------------------------------------------------------------
    # Folder whose sub-folders are bids (Bid1, Bid2, ...). Original files stay in place.
    bids_root: Path = PROJECT_ROOT
    # Glob used by `index --all` to discover bid folders under bids_root.
    bid_folder_glob: str = "Bid*"
    data_dir: Path = PROJECT_ROOT / "data"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    docling_artifacts_dir: Path = PROJECT_ROOT / "data" / "models" / "docling"
    # Extraction outputs written from the UI (<bid>.json, .report.md, .trace.jsonl, .cli.log).
    output_dir: Path = PROJECT_ROOT / "output"

    # ---- Ingestion ---------------------------------------------------------------
    pdf_parser_backend: str = "docling"  # "docling" (layout+tables) or "pymupdf" (fast)
    # OCR bitmap regions / scanned pages with RapidOCR (Docling). ~2 s/page on CPU.
    pdf_ocr_enabled: bool = True
    pdf_table_structure: bool = True
    # Docling converts PDFs in page batches so background jobs can report page progress
    # and be cancelled between batches.
    pdf_batch_pages: int = 10
    # Uploads above either limit are flagged in the UI before indexing (warning + OCR choice).
    large_pdf_pages: int = 50
    large_pdf_images: int = 200
    # Time estimate shown in that warning. Measured on the dev laptop (CPU): the 62-page
    # Bid1 RFP took ~160 s with OCR = ~2.6 s/page. Image-heavy pages are slower.
    ocr_seconds_per_page: float = 2.6
    # Fallback OCR (only when OCR is on): pages that are scanned (no text layer) or mostly
    # image, but for which the parser returned < ocr_fallback_min_chars, are rendered with
    # PyMuPDF and read with RapidOCR directly (same local models Docling uses).
    ocr_fallback_enabled: bool = True
    ocr_fallback_min_chars: int = 30
    ocr_fallback_image_coverage: float = 0.3  # image area / page area that counts as "mostly image"
    ocr_fallback_dpi: int = 200
    ocr_fallback_min_score: float = 0.5  # drop recognised segments below this confidence
    max_upload_mb: int = 25
    allowed_extensions: StrList = [".pdf", ".html", ".htm"]

    # ---- Chunking ----------------------------------------------------------------
    chunk_size: int = 1200  # characters (~250-300 tokens)
    chunk_overlap: int = 200  # characters carried into the next chunk
    max_table_chunk_size: int = 2400  # tables up to this size stay whole
    min_chunk_chars: int = 40

    # ---- Embeddings / vector store -----------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Instruction prepended to queries (BGE-style models). Empty for models without one.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_device: str = "cpu"  # "cpu", "cuda" or "auto"
    embedding_batch_size: int = 32
    collection_name: str = "rfp_chunks"

    # ---- Retrieval ---------------------------------------------------------------
    default_top_k: int = 5
    candidate_pool: int = 30  # per-retriever candidates before fusion
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    query_expansion: bool = True
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    # "replace": order by cross-encoder score only; "rrf": fuse reranker rank with hybrid rank
    rerank_fusion: str = "replace"
    rerank_weight: float = 1.0
    # extra RRF vote for addendum chunks when the query asks for a revised/latest value (0 = off)
    addendum_intent_weight: float = 0.0
    default_search_mode: str = "hybrid_rerank"  # vector | bm25 | hybrid | hybrid_rerank

    # ---- LLM (OpenRouter, free models only) --------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_models: StrList = Field(default_factory=lambda: list(DEFAULT_FREE_MODELS))
    llm_default_model: str = DEFAULT_FREE_MODELS[0]
    # Explicit, ordered fallback list (must also be free). Empty = no fallback.
    llm_fallback_models: StrList = Field(default_factory=list)
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048  # reasoning models count thinking tokens against this
    llm_reasoning_effort: str = "low"  # "", "low", "medium", "high" (OpenRouter `reasoning`)
    # Verify zero pricing via the OpenRouter catalogue before calling a model.
    llm_verify_free: bool = True
    ask_context_chunks: int = 6  # top results of the main search mode given to the LLM
    ask_hybrid_extra: int = 4  # + plain-hybrid results (recall safety net for the reranker)
    ask_addendum_extra: int = 3  # + addendum passages for "revised/latest" questions
    ask_max_context: int = 10  # cap on passages sent to the LLM
    # Model preselected for UI extraction runs (the one the Bid1 / Bid2 outputs were made with).
    extract_default_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    app_referer: str = "http://localhost:8501"
    app_title: str = "RFP Intelligence Platform"

    # ---- Services ----------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_url: str = "http://127.0.0.1:8000"
    log_level: str = "INFO"

    @field_validator("llm_models", "llm_fallback_models", "allowed_extensions", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return json.loads(v)
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator(
        "bids_root", "data_dir", "uploads_dir", "index_dir", "docling_artifacts_dir", "output_dir", mode="after"
    )
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


# Third-party loggers that are chatty at INFO (per-page Docling stages, HTTP calls, ...).
_QUIET_LOGGERS = (
    "httpx", "httpcore", "urllib3", "chromadb", "docling", "docling_core", "docling_ibm_models",
    "sentence_transformers", "transformers", "huggingface_hub", "filelock", "fsspec", "PIL",
    "RapidOCR", "rapidocr", "onnxruntime", "matplotlib", "multipart", "datasets",
)


def configure_logging(level: str | None = None) -> None:
    """Console logging for the app's own loggers (idempotent; safe to call repeatedly)."""
    import logging
    import sys

    root = logging.getLogger()
    if not any(getattr(h, "_rfp_console", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._rfp_console = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel((level or get_settings().log_level).upper())
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
