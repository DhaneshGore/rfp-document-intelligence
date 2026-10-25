from pathlib import Path

import pytest

from ingestion.cleaner import clean_blocks, join_wrapped_lines, normalize_text, remove_repeated_furniture
from ingestion.html_parser import HtmlParser
from ingestion.loader import DocumentLoader, UnsupportedFileError
from ingestion.pdf_parser import PdfParser, render_table_rows
from schemas import Block, BlockType, DocType
from tests.conftest import BID_HTML, make_pdf


# ----------------------------------------------------------------------- HTML
def test_html_extracts_fields_tables_and_drops_chrome():
    blocks, title = HtmlParser().parse_html(BID_HTML)
    texts = [b.text for b in blocks]
    joined = "\n".join(texts)
    assert title == "Widget Supply - Bid Information"
    assert "Solicitation Number: RFQ-555" in texts
    assert "Closing Date: 04/15/2024 02:00 PM EST" in texts
    assert any(b.type == BlockType.HEADING and b.text == "Dates" for b in blocks)
    table = next(b for b in blocks if b.type == BlockType.TABLE)
    assert table.table_header == "Item | Qty"
    assert table.table_rows == ["Blue widget WX-100 | 40"]
    for noise in ("Logout", "cookies", "should not appear", "Copyright footer"):
        assert noise not in joined
    assert all(b.page is None for b in blocks)  # HTML has no page numbers


def test_html_content_inside_form_is_kept():
    html = "<html><body><main><form><h2>Notice</h2><p>Bids close Friday at noon.</p></form></main></body></html>"
    blocks, _ = HtmlParser().parse_html(html)
    assert "Bids close Friday at noon." in [b.text for b in blocks]


def test_render_table_rows_collapses_merged_cells():
    rows = render_table_rows([["Section 1", "Section 1"], ["Due", "06/10/2024"], ["", ""]])
    assert rows == ["Section 1", "Due | 06/10/2024"]


# ----------------------------------------------------------------------- PDF
def test_pdf_pymupdf_preserves_pages_and_reports_empty(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", ["First page text about bonds.", "", "Third page warranty."])
    res = PdfParser(backend="pymupdf", ocr=False).parse(pdf)
    pages = {b.page for b in res.blocks}
    assert pages == {1, 3}
    assert res.num_pages == 3
    assert res.empty_pages == [2]
    assert any("no text layer" in w for w in res.warnings)


def test_docling_failure_falls_back_to_pymupdf(tmp_path, settings, monkeypatch):
    pdf = make_pdf(tmp_path / "doc.pdf", ["Fallback content here."])
    parser = PdfParser(backend="docling", ocr=False)
    monkeypatch.setattr(parser, "_parse_docling", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    res = parser.parse(pdf)
    assert res.parser == "pymupdf"
    assert any("Docling failed" in w for w in res.warnings)
    assert "Fallback content here." in " ".join(b.text for b in res.blocks)


# ----------------------------------------------------------------------- cleaning
def test_normalize_text_whitespace_hyphenation_and_fillers():
    raw = "Pre-\nproposal   meeting  on\tFriday ..........  ______\n\n\n\nNext"
    out = normalize_text(raw)
    assert "Pre-proposal" in out
    assert "....." not in out and "____" not in out
    assert "\n\n\n" not in out
    assert "meeting on Friday" in out


def test_join_wrapped_lines_keeps_list_items():
    text = "This sentence was\nwrapped by layout.\n- bullet one\n- bullet two"
    assert join_wrapped_lines(text) == "This sentence was wrapped by layout.\n- bullet one\n- bullet two"


def test_repeated_headers_removed_but_first_kept():
    blocks = []
    body_numbers = [7, 3, 9, 1, 4]  # body lines that differ only by a non-increasing number
    for p in range(1, 6):
        blocks += [
            Block(type=BlockType.TEXT, text="ACME RFP 123 Widget Supply", page=p),
            Block(type=BlockType.TEXT, text=f"Page {p} of 5", page=p),
            Block(type=BlockType.TEXT, text=f"Confidential - rev 2 - sheet {p + 10}", page=p),
            Block(type=BlockType.TEXT, text=f"Requirement {body_numbers[p - 1]} applies to all vendors", page=p),
        ]
    kept, removed = remove_repeated_furniture(blocks, 5)
    texts = [b.text for b in kept]
    assert texts.count("ACME RFP 123 Widget Supply") == 1
    assert not any(t.startswith("Page ") for t in texts)
    assert sum(t.startswith("Confidential") for t in texts) == 1  # page counter -> furniture
    assert sum(t.startswith("Requirement") for t in texts) == 5  # real content kept
    assert removed


def test_clean_blocks_drops_empty():
    out = clean_blocks([Block(type=BlockType.TEXT, text="   "), Block(type=BlockType.TEXT, text=" ok ")])
    assert [b.text for b in out] == ["ok"]


# ----------------------------------------------------------------------- loader / failures
def test_loader_builds_metadata(bid_folder, settings):
    loader = DocumentLoader(PdfParser(backend="pymupdf"))
    doc = loader.load(bid_folder / "Addendum 1 RFP-555.pdf", "BidA")
    md = doc.metadata
    assert md.doc_type == DocType.ADDENDUM
    assert md.addendum_number == 1
    assert md.bid_id == "BidA"
    assert md.num_pages == 1
    assert len(md.content_hash) == 64
    html = loader.load(bid_folder / "RFQ-555 Bid Information.html", "BidA")
    assert html.metadata.doc_type == DocType.BID_PAGE
    assert html.metadata.document_date == "2024-03-01"  # "Publication" label


def test_loader_missing_file(settings):
    with pytest.raises(FileNotFoundError):
        DocumentLoader(PdfParser(backend="pymupdf")).load(Path("does/not/exist.pdf"), "BidA")


def test_loader_empty_file(tmp_path, settings):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        DocumentLoader(PdfParser(backend="pymupdf")).load(p, "BidA")


def test_loader_unsupported_type(tmp_path, settings):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(UnsupportedFileError):
        DocumentLoader(PdfParser(backend="pymupdf")).load(p, "BidA")


def test_loader_corrupt_pdf_raises(tmp_path, settings):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(Exception):
        DocumentLoader(PdfParser(backend="pymupdf")).load(p, "BidA")


def test_loader_blank_pdf_flagged_empty(tmp_path, settings):
    p = make_pdf(tmp_path / "blank.pdf", ["", ""])
    doc = DocumentLoader(PdfParser(backend="pymupdf", ocr=False)).load(p, "BidA")
    assert doc.is_empty
    assert any("No extractable text" in w for w in doc.warnings)
