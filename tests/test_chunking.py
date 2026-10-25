import pytest

from ingestion.chunker import Chunker
from schemas import Block, BlockType, DocType, DocumentMetadata, ParsedDocument


def _doc(blocks, doc_type=DocType.RFP, addendum=None):
    md = DocumentMetadata(
        bid_id="BidA", file_name="rfp.pdf", source_path="/x/rfp.pdf", doc_type=doc_type,
        addendum_number=addendum, document_date="2024-03-01", content_hash="h" * 64,
    )
    return ParsedDocument(metadata=md, blocks=blocks)


def T(text, page=1):
    return Block(type=BlockType.TEXT, text=text, page=page)


def H(text, page=1, level=1):
    return Block(type=BlockType.HEADING, text=text, page=page, level=level)


def test_headings_start_new_chunks_and_prefix_section(settings):
    doc = _doc([H("BID BOND"), T("A bid bond of 5% is required."), H("WARRANTY", page=2), T("Three year warranty.", page=2)])
    chunks = Chunker().chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].section == "BID BOND" and chunks[0].text.startswith("BID BOND\n")
    assert "warranty" not in chunks[0].text.lower()
    assert chunks[1].section == "WARRANTY" and chunks[1].page_number == 2


def test_nested_heading_path(settings):
    doc = _doc([H("Specs", level=1), H("Tier 1 Laptop", level=2), T("8GB RAM")])
    assert Chunker().chunk(doc)[0].section == "Specs > Tier 1 Laptop"


def test_repeated_running_heading_does_not_split(settings):
    doc = _doc([H("SCOPE"), T("Part one of scope."), H("SCOPE", page=2), T("Part two of scope.", page=2)])
    chunks = Chunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1 and chunks[0].page_end == 2


def test_chunk_size_and_overlap(settings):
    sentences = [T(f"Sentence number {i} describes a requirement in detail.") for i in range(40)]
    chunks = Chunker(chunk_size=300, chunk_overlap=60).chunk(_doc(sentences))
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 60 for c in chunks)
    # overlap: the start of each chunk repeats the tail of the previous one
    for a, b in zip(chunks, chunks[1:]):
        first_line = b.text.split("\n")[0]
        assert first_line in a.text


def test_long_paragraph_is_split_on_sentences(settings):
    para = " ".join(f"Clause {i} applies to all vendors." for i in range(60))
    chunks = Chunker(chunk_size=250, chunk_overlap=0).chunk(_doc([T(para)]))
    assert len(chunks) > 5
    assert all(len(c.text) <= 250 for c in chunks)


def test_small_table_kept_whole(settings):
    table = Block(type=BlockType.TABLE, text="Item | Qty\nWidget | 4", page=3,
                  table_header="Item | Qty", table_rows=["Widget | 4"])
    chunks = Chunker().chunk(_doc([T("Intro"), table, T("After")]))
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1 and "Widget | 4" in tables[0].text and tables[0].page_number == 3
    assert len(chunks) == 3  # table never merged with surrounding text


def test_large_table_split_with_header_repeated(settings):
    rows = [f"Line {i} | Laptop model X{i} with long description text | {i * 10}" for i in range(60)]
    table = Block(type=BlockType.TABLE, text="Line | Description | Qty\n" + "\n".join(rows), page=1,
                  table_header="Line | Description | Qty", table_rows=rows)
    chunks = Chunker(max_table_chunk_size=600).chunk(_doc([table]))
    assert len(chunks) > 1
    for c in chunks:
        assert "Line | Description | Qty" in c.text
        assert len(c.text) <= 600 + 50
    body = "\n".join(c.text for c in chunks)
    assert all(r in body for r in rows)  # no row lost


def test_metadata_propagated_and_ids_deterministic(settings):
    doc = _doc([T("Addendum text")], doc_type=DocType.ADDENDUM, addendum=2)
    a, b = Chunker().chunk(doc), Chunker().chunk(doc)
    c = a[0]
    assert (c.bid_id, c.file_name, c.doc_type, c.addendum_number, c.document_date) == (
        "BidA", "rfp.pdf", DocType.ADDENDUM, 2, "2024-03-01"
    )
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert c.chunk_id.startswith(c.doc_id)


def test_invalid_overlap_rejected(settings):
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)
