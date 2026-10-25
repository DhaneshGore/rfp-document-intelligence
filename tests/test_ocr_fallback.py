"""Fallback OCR (PyMuPDF render + RapidOCR) for pages the main parser left empty."""
from pathlib import Path

import pytest

from config import PROJECT_ROOT
from ingestion.ocr_fallback import FallbackOcr, PageInfo, page_profile, pages_needing_ocr, segments_to_blocks
from ingestion.pdf_parser import PdfParser
from jobcontrol import JobCancelled, JobControl, set_current
from schemas import Block, BlockType

RAPIDOCR_MODELS = PROJECT_ROOT / "data" / "models" / "docling" / "RapidOcr"
TEXT = "Proposals are due on April 15, 2024 at 2:00 PM local time."
SCANNED = "SCANNED NOTICE\nDelivery address: Warehouse 7, Dock B\nContact the purchasing office for details."


def make_mixed_pdf(path: Path, pages: list[tuple[str, bool]]) -> Path:
    """(text, scanned) per page. Scanned pages are an image of the text, no text layer."""
    import fitz

    src = fitz.open()
    for text, _ in pages:
        page = src.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=14)
    out = fitz.open()
    for i, (_, scanned) in enumerate(pages):
        if scanned:
            pix = src[i].get_pixmap(dpi=150)
            page = out.new_page(width=src[i].rect.width, height=src[i].rect.height)
            page.insert_image(page.rect, pixmap=pix)
        else:
            out.insert_pdf(src, from_page=i, to_page=i)
    out.save(path)
    out.close()
    src.close()
    return path


@pytest.fixture
def parser(settings):
    settings.ocr_fallback_enabled = True
    return PdfParser(backend="pymupdf", ocr=True)


def fake_ocr(calls):
    def ocr_page(path, page_no):
        calls.append(page_no)
        return [Block(type=BlockType.TEXT, text=f"recognised text from scanned page {page_no}", page=page_no)]
    return ocr_page


# ----------------------------------------------------------------------- unit
def test_segments_grouped_into_lines_and_paragraphs():
    segs = [  # left, top, right, bottom, text (pixels)
        (300, 10, 400, 30, "DHL-42"), (10, 10, 90, 30, "Part No:"),  # same line, wide gap
        (10, 40, 200, 60, "second line"),  # same paragraph
        (10, 120, 200, 140, "new paragraph"),  # big vertical gap
    ]
    blocks = segments_to_blocks(segs, page=4)
    assert [b.text for b in blocks] == ["Part No: | DHL-42\nsecond line", "new paragraph"]
    assert {b.page for b in blocks} == {4}
    assert segments_to_blocks([], page=1) == []


def test_pages_needing_ocr_selects_only_empty_scanned_or_image_pages():
    profile = {
        1: PageInfo(text_chars=500, image_coverage=0.0),  # normal text page, parsed
        2: PageInfo(text_chars=0, image_coverage=1.0),  # scanned, parser got nothing
        3: PageInfo(text_chars=300, image_coverage=0.0),  # text layer, parser missed it: not OCR
        4: PageInfo(text_chars=40, image_coverage=0.8),  # mostly image, little text
        5: PageInfo(text_chars=0, image_coverage=1.0),  # scanned but parser already read it
    }
    blocks = [Block(type=BlockType.TEXT, text="x" * 500, page=1), Block(type=BlockType.TEXT, text="y" * 80, page=5)]
    assert pages_needing_ocr(blocks, profile, min_chars=30, min_coverage=0.3) == [2, 4]


def test_page_profile_reads_text_layer_and_image_coverage(tmp_path):
    pdf = make_mixed_pdf(tmp_path / "m.pdf", [(TEXT, False), (SCANNED, True)])
    prof = page_profile(pdf)
    assert prof[1].text_chars > 20 and prof[1].image_coverage == 0
    assert prof[2].text_chars == 0 and prof[2].image_coverage > 0.9


# ----------------------------------------------------------------------- parser integration (fake engine)
def test_mixed_pdf_only_scanned_pages_are_ocred_in_page_order(parser, tmp_path):
    pdf = make_mixed_pdf(tmp_path / "mixed.pdf", [(TEXT, False), (SCANNED, True), (TEXT, False), (SCANNED, True)])
    calls = []
    parser.fallback_ocr.ocr_page = fake_ocr(calls)
    res = parser.parse(pdf)
    assert calls == [2, 4] and res.ocr_fallback_pages == [2, 4]
    assert [b.page for b in res.blocks] == sorted(b.page for b in res.blocks)  # document order kept
    assert {b.page for b in res.blocks} == {1, 2, 3, 4} and res.empty_pages == []
    assert any("Fallback OCR" in w and "[2, 4]" in w for w in res.warnings)


def test_docling_failure_then_pymupdf_then_fallback_ocr(settings, tmp_path, monkeypatch):
    settings.ocr_fallback_enabled = True
    parser = PdfParser(backend="docling", ocr=True)
    monkeypatch.setattr(parser, "_parse_docling", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    calls = []
    parser.fallback_ocr.ocr_page = fake_ocr(calls)
    res = parser.parse(make_mixed_pdf(tmp_path / "s.pdf", [(TEXT, False), (SCANNED, True)]))
    assert res.parser == "pymupdf" and calls == [2] and res.ocr_fallback_pages == [2]
    assert any("Docling failed" in w for w in res.warnings)


def test_ocr_error_on_one_page_is_reported_and_other_pages_continue(parser, tmp_path):
    pdf = make_mixed_pdf(tmp_path / "e.pdf", [(SCANNED, True), (SCANNED, True), (TEXT, False)])
    ok = fake_ocr([])

    def flaky(path, page_no):
        if page_no == 1:
            raise RuntimeError("onnx exploded")
        return ok(path, page_no)

    parser.fallback_ocr.ocr_page = flaky
    res = parser.parse(pdf)
    assert res.ocr_fallback_pages == [2] and res.empty_pages == [1]
    assert any(w.startswith("Fallback OCR failed on page(s) [1]") and "onnx exploded" in w for w in res.warnings)


def test_missing_ocr_models_stop_fallback_with_one_warning(parser, tmp_path):
    parser.fallback_ocr = FallbackOcr(models_dir=tmp_path / "no-models")
    res = parser.parse(make_mixed_pdf(tmp_path / "x.pdf", [(SCANNED, True), (SCANNED, True)]))
    failed = [w for w in res.warnings if w.startswith("Fallback OCR failed")]
    assert len(failed) == 1 and "[1, 2]" in failed[0] and "FileNotFoundError" in failed[0]
    assert res.ocr_fallback_pages == [] and res.empty_pages == [1, 2]


def test_fallback_not_used_when_parser_text_is_longer(parser, tmp_path):
    pdf = make_mixed_pdf(tmp_path / "t.pdf", [(SCANNED, True)])
    parser.fallback_ocr.ocr_page = lambda path, p: [Block(type=BlockType.TEXT, text="", page=p)]
    res = parser.parse(pdf)
    assert res.ocr_fallback_pages == [] and not any("read page(s)" in w for w in res.warnings)


def test_ocr_disabled_or_fallback_disabled_skips_fallback(parser, settings, tmp_path):
    pdf = make_mixed_pdf(tmp_path / "d.pdf", [(TEXT, False), (SCANNED, True)])
    calls = []
    parser.fallback_ocr.ocr_page = fake_ocr(calls)
    res = parser.parse(pdf, ocr=False)
    assert calls == [] and res.empty_pages == [2] and res.ocr_fallback_pages == []
    assert any("index with OCR to read them" in w for w in res.warnings)
    settings.ocr_fallback_enabled = False
    assert parser.parse(pdf).empty_pages == [2] and calls == []


def test_cancel_stops_fallback_between_pages(parser, tmp_path):
    pdf = make_mixed_pdf(tmp_path / "c.pdf", [(SCANNED, True), (SCANNED, True), (SCANNED, True)])
    calls, details = [], []
    ctrl = JobControl(on_detail=details.append)
    inner = fake_ocr(calls)

    def ocr_then_cancel(path, page_no):
        ctrl.cancel_event.set()  # user clicks Cancel while page 1 is being read
        return inner(path, page_no)

    parser.fallback_ocr.ocr_page = ocr_then_cancel
    set_current(ctrl)
    try:
        with pytest.raises(JobCancelled):
            parser.parse(pdf)
    finally:
        set_current(None)
    assert calls == [1] and details == ["c.pdf: fallback OCR page 1 (1/3)"]


def test_cancelled_fallback_keeps_previous_index_version(engine, settings, tmp_path):
    settings.ocr_fallback_enabled = True
    pdf = make_mixed_pdf(tmp_path / "doc.pdf", [(TEXT, False)])
    engine.indexing.index_file(pdf, "BidX")
    old = engine.manifest.get(pdf)
    make_mixed_pdf(pdf, [(TEXT, False), (SCANNED, True)])
    p = engine.indexing.loader.pdf
    p.ocr = True
    ctrl = JobControl()
    p.fallback_ocr.ocr_page = lambda path, n: ctrl.cancel_event.set() or []
    set_current(ctrl)
    try:
        with pytest.raises(JobCancelled):
            engine.indexing.index_file(pdf, "BidX")
    finally:
        set_current(None)
    assert engine.manifest.get(pdf)["content_hash"] == old["content_hash"]
    assert engine.vector_store.doc_chunk_count(old["doc_id"]) > 0


# ----------------------------------------------------------------------- real RapidOCR engine
@pytest.fixture
def real_parser(parser):
    if not RAPIDOCR_MODELS.exists():
        pytest.skip("local RapidOCR models not available")
    parser.fallback_ocr = FallbackOcr(models_dir=RAPIDOCR_MODELS)
    return parser


@pytest.mark.slow
def test_real_ocr_reads_scanned_pdf(real_parser, tmp_path):
    res = real_parser.parse(make_mixed_pdf(tmp_path / "scan.pdf", [(SCANNED, True)]))
    text = " ".join(b.text for b in res.blocks)
    assert res.ocr_fallback_pages == [1] and {b.page for b in res.blocks} == {1}
    for word in ("SCANNED", "NOTICE", "Warehouse", "Dock", "purchasing"):
        assert word in text, text


@pytest.mark.slow
def test_real_ocr_mixed_pdf_keeps_text_layer_and_page_numbers(real_parser, tmp_path):
    pdf = make_mixed_pdf(tmp_path / "mix.pdf", [(TEXT, False), (SCANNED, True), (TEXT, False)])
    res = real_parser.parse(pdf)
    assert res.ocr_fallback_pages == [2]
    by_page = {p: " ".join(b.text for b in res.blocks if b.page == p) for p in (1, 2, 3)}
    assert "April 15, 2024" in by_page[1] and "April 15, 2024" in by_page[3]  # text layer, untouched
    assert "Warehouse" in by_page[2] and "April" not in by_page[2]
