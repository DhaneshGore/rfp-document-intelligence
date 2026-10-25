"""Shared fixtures. Tests run without API keys, network or model downloads:
embeddings use a deterministic hashing embedder and PDFs use the PyMuPDF backend
(the Docling path is covered by tests marked `slow`)."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, fast, offline."""

    name = "fake-hash-embedder"
    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ReverseReranker:
    """Fake cross-encoder: scores by passage length (deterministic, easy to assert)."""

    name = "fake-reranker"

    def score(self, query, passages):
        return [float(len(p)) for p in passages]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    env = {
        "BIDS_ROOT": tmp_path / "bids",
        "DATA_DIR": tmp_path / "data",
        "UPLOADS_DIR": tmp_path / "data" / "uploads",
        "INDEX_DIR": tmp_path / "data" / "index",
        "OUTPUT_DIR": tmp_path / "output",  # never the project's real output/ folder
        "RERANKER_ENABLED": "false",
        "OPENROUTER_API_KEY": "",
        "PDF_PARSER_BACKEND": "pymupdf",
        "OCR_FALLBACK_ENABLED": "false",  # enabled explicitly in test_ocr_fallback.py
        "LLM_FALLBACK_MODELS": "",
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "80",
        "MAX_TABLE_CHUNK_SIZE": "600",
        "MAX_UPLOAD_MB": "1",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    (tmp_path / "bids").mkdir()
    config.get_settings.cache_clear()
    s = config.get_settings()
    yield s
    config.get_settings.cache_clear()


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Create a simple text PDF; an empty string makes a blank page."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()
    return path


BID_HTML = """<!doctype html><html><head><title>Widget Supply - Bid Information</title></head>
<body>
<nav class="main-nav"><a>Solicitations</a><a>Search</a><a>Logout</a></nav>
<div id="cookie-banner">We use cookies to improve things.</div>
<main class="content">
  <h1>RFQ-555 - Widget Supply</h1>
  <h3>Basic Information</h3>
  <div class="field"><span class="field-label">Solicitation Number</span><div class="body"><p>RFQ-555</p></div></div>
  <div class="field"><span class="field-label">Issuing Organization</span><div class="body"><p>City of Springfield</p></div></div>
  <h3>Dates</h3>
  <div class="field"><span class="field-label">Publication</span><div class="body"><p>03/01/2024 10:00 AM EST</p></div></div>
  <div class="field"><span class="field-label">Closing Date</span><div class="body"><p>04/15/2024 02:00 PM EST</p></div></div>
  <table><tr><th>Item</th><th>Qty</th></tr><tr><td>Blue widget WX-100</td><td>40</td></tr></table>
  <script>var x = "should not appear";</script>
</main>
<footer class="site-footer">Copyright footer text</footer>
</body></html>"""


@pytest.fixture
def bid_folder(settings) -> Path:
    """A small bid folder: RFP PDF, addendum PDF, specs PDF, affidavit PDF, HTML page."""
    d = settings.bids_root / "BidA"
    d.mkdir()
    make_pdf(d / "RFP-555 Widget Supply.pdf", [
        "Request for Proposal RFP-555\nIssue Date: 03/01/2024\n\nSUBMISSION\n"
        "Proposals are due April 1, 2024 at 2:00 PM. Submit electronically through the portal.",
        "BID BOND\nA bid bond of 5% of the total bid amount is required with each proposal.\n\n"
        "WARRANTY\nAll widgets require a three year manufacturer warranty.",
    ])
    make_pdf(d / "Addendum 1 RFP-555.pdf", [
        "ADDENDUM No. 1\nThe due date is extended to April 15, 2024 at 2:00 PM."
    ])
    make_pdf(d / "Widget_Specs.pdf", [
        "Technical specifications\nModel WX-100, part number 210-ABCD, 16 GB memory, 512 GB SSD."
    ])
    make_pdf(d / "Vendor_Affidavit.pdf", [
        "AFFIDAVIT\nI hereby affirm that the business is registered in the State."
    ])
    (d / "RFQ-555 Bid Information.html").write_text(BID_HTML, encoding="utf-8")
    return d


@pytest.fixture
def engine(settings):
    from ingestion.loader import DocumentLoader
    from ingestion.pdf_parser import PdfParser
    from services.container import build_engine

    return build_engine(
        settings,
        embedder=FakeEmbedder(),
        reranker=None,
        loader=DocumentLoader(PdfParser(backend="pymupdf")),
    )
