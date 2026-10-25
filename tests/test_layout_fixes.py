"""Regression tests for reading-order repair and borderless code-table recovery."""
import fitz

from ingestion.pdf_parser import PdfParser, merge_recovered_tables, reading_order, recover_code_tables
from schemas import Block, BlockType


def T(text, page=1):
    return Block(type=BlockType.TEXT, text=text, page=page)


def test_reading_order_fixes_swapped_list_items():
    # Docling-style output: list items grouped before the interleaved answers
    blocks = [T("Q1?"), T("Q2?"), T("A1."), T("A2.")]
    boxes = [(90, 100, 400, 110), (90, 140, 400, 150), (126, 120, 400, 130), (126, 160, 400, 170)]
    out = reading_order(blocks, boxes, [False] * 4)
    assert [b.text for b in out] == ["Q1?", "A1.", "Q2?", "A2."]


def test_reading_order_keeps_multi_column_pages():
    blocks = [T("left top"), T("left bottom"), T("right top")]
    boxes = [(50, 100, 280, 110), (50, 300, 280, 310), (320, 100, 550, 110)]
    out = reading_order(blocks, boxes, [False] * 3)
    assert [b.text for b in out] == ["left top", "left bottom", "right top"]


def test_reading_order_ignores_furniture_and_missing_boxes():
    blocks = [T("body 2"), T("header"), T("body 1"), T("p2", page=2)]
    boxes = [(50, 300, 500, 310), (400, 300, 550, 310), (50, 100, 500, 110), None]
    out = reading_order(blocks, boxes, [False, True, False, False])
    # the side-by-side furniture does not make the page "multi-column"
    assert [b.text for b in out] == ["body 1", "body 2", "header", "p2"]


SKUS = ["210-BLYZ", "379-BFNZ", "619-ARSB", "658-BCSB", "338-CNRG", "338-CNRK"]
DESCS = [
    ("Laptop XCTO Base",),
    ("Intel Core Ultra 5 125U processor (12 MB cache,", "12 cores, up to 4.3 GHz Turbo)"),
    ("Windows 11 Pro",),
    ("No Office License",),
    ("16 GB: 2 x 8 GB, DDR5",),
    ("256 GB, M.2 2230, NVMe, SSD",),
]


def make_spec_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((22, 92), "Quote 5550 Laptop", fontsize=10)
    page.insert_text((22, 112), "Description", fontsize=9)
    page.insert_text((320, 108), "SKU", fontsize=9)
    y = 130
    for sku, lines in zip(SKUS, DESCS):
        page.insert_text((300, y), sku, fontsize=9)
        for k, line in enumerate(lines):
            page.insert_text((22, y + 11 * k), line, fontsize=9)
        y += 11 * len(lines) + 9
    doc.save(path)
    doc.close()
    return path


def test_recover_code_table_pairs_sku_with_description(tmp_path):
    pdf = make_spec_pdf(tmp_path / "specs.pdf")
    table = recover_code_tables(pdf)[1]
    assert table.table_header == "Description | SKU"
    assert table.table_rows[0] == "Laptop XCTO Base | 210-BLYZ"
    assert table.table_rows[1] == (
        "Intel Core Ultra 5 125U processor (12 MB cache, 12 cores, up to 4.3 GHz Turbo) | 379-BFNZ"
    )
    assert table.table_rows[-1] == "256 GB, M.2 2230, NVMe, SSD | 338-CNRK"
    assert len(table.table_rows) == len(SKUS)


def test_no_code_table_on_ordinary_text(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 800), "Ordinary paragraph about bid bonds. " * 40)
    doc.save(tmp_path / "plain.pdf")
    assert recover_code_tables(tmp_path / "plain.pdf") == {}


def test_parser_replaces_fragments_with_recovered_table(tmp_path, settings):
    pdf = make_spec_pdf(tmp_path / "specs.pdf")
    res = PdfParser(backend="pymupdf", ocr=False).parse(pdf)
    tables = [b for b in res.blocks if b.type == BlockType.TABLE]
    assert any("Windows 11 Pro | 619-ARSB" in t.text for t in tables)
    loose = [b.text for b in res.blocks if b.type != BlockType.TABLE]
    assert not any(t.strip() == "210-BLYZ" for t in loose)  # code no longer a loose fragment


def test_merge_skips_page_with_existing_table():
    existing = Block(type=BlockType.TABLE, text="a | 210-BLYZ\nb | 379-BFNZ", page=1,
                     table_rows=["a | 210-BLYZ", "b | 379-BFNZ"])
    recovered = Block(type=BlockType.TABLE, text="x", page=1, table_rows=["a | 210-BLYZ", "b | 379-BFNZ"])
    assert merge_recovered_tables([existing], {1: recovered}) == [existing]
