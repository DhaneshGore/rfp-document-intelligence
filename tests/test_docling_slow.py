"""Docling integration on a real assignment document (slow: loads layout models).

Skipped automatically when Docling or its local models are unavailable.
Run only these with:  pytest -m slow
"""
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.slow

MERCURY = next(iter(sorted(PROJECT_ROOT.glob("Bid*/Mercury_Affidavit.pdf"))), None)
PORFP = next(iter(sorted(PROJECT_ROOT.glob("Bid*/PORFP*.pdf"))), None)


def _require(path):
    pytest.importorskip("docling")
    if path is None:
        pytest.skip("assignment PDF not present")
    models = PROJECT_ROOT / "data" / "models" / "docling"
    if not models.exists():
        pytest.skip("Docling models not downloaded (run `python main.py index --all` once)")


def test_docling_parses_real_affidavit(monkeypatch):
    _require(MERCURY)
    from ingestion.loader import DocumentLoader
    from ingestion.pdf_parser import PdfParser
    from schemas import DocType

    monkeypatch.setenv("DOCLING_ARTIFACTS_DIR", str(PROJECT_ROOT / "data" / "models" / "docling"))
    doc = DocumentLoader(PdfParser(backend="docling", ocr=False)).load(Path(MERCURY), "Bid2")
    assert doc.metadata.parser == "docling"
    assert doc.metadata.doc_type == DocType.AFFIDAVIT
    text = " ".join(b.text for b in doc.blocks)
    assert "mercury" in text.lower()
    assert {b.page for b in doc.blocks} == {1}


def test_docling_extracts_tables_from_porfp():
    _require(PORFP)
    from ingestion.pdf_parser import PdfParser
    from schemas import BlockType

    res = PdfParser(backend="docling", ocr=False).parse(Path(PORFP))
    tables = [b for b in res.blocks if b.type == BlockType.TABLE]
    assert tables, "expected Docling to detect the PORFP form tables"
    assert any("Tamaira Hawkins" in t.text for t in tables)
    assert all(t.page is not None for t in tables)
