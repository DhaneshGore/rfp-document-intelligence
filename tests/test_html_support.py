"""HTML support: encodings, UTF-16 uploads, JavaScript-rendered pages, HTML doc types,
section-based citations for page-less passages."""
import pytest

from ingestion.encoding import decode_html, detect_html_encoding
from ingestion.html_parser import HtmlParser
from ingestion.metadata import classify_doc_type, extract_addendum_number
from schemas import Citation, DocType, SearchFilters
from services.uploads import UploadError, store_upload, validate_content
from tests.conftest import BID_HTML

PAGE = ('<!DOCTYPE html><html><head>{meta}<title>Notice</title></head><body><main>'
        '<h1>Library Furniture</h1><p>Walkthrough at Café Hall; buyer “Maria López”, see § 4.2.</p>'
        '</main></body></html>')
EXPECTED = "Walkthrough at Café Hall; buyer “Maria López”, see § 4.2."


def _page(meta: str = "") -> str:
    return PAGE.format(meta=meta)


# ----------------------------------------------------------------------- encodings
@pytest.mark.parametrize("raw,enc", [
    (_page('<meta charset="utf-8">').encode("utf-8"), "utf-8"),
    (b"\xef\xbb\xbf" + _page().encode("utf-8"), "utf-8-sig"),
    (_page('<meta charset="windows-1252">').encode("cp1252"), "cp1252"),
    (_page('<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">').encode("cp1252"), "cp1252"),
    (_page().encode("cp1252"), "cp1252"),  # no declaration, not valid UTF-8
    (_page('<meta charset="utf-8">').encode("cp1252"), "cp1252"),  # wrong declaration
    (_page().encode("utf-16"), "utf-16"),  # with BOM
    (_page().encode("utf-16-le"), "utf-16-le"),  # without BOM
    (_page().encode("utf-16-be"), "utf-16-be"),
])
def test_html_encoding_detected_and_text_intact(raw, enc, tmp_path):
    assert detect_html_encoding(raw) == enc
    assert "�" not in decode_html(raw)
    f = tmp_path / "p.htm"
    f.write_bytes(raw)
    blocks, title = HtmlParser().parse(f)
    assert EXPECTED in [b.text for b in blocks] and title == "Notice"


def test_utf16_html_upload_accepted_and_binary_still_rejected(settings):
    validate_content("p.html", _page().encode("utf-16"))
    validate_content("p.htm", _page().encode("utf-16-le"))
    path, _ = store_upload("EncBid", "page.html", _page().encode("utf-16"))
    assert path.exists()
    for bad in (b"\x00\x01binary", b"MZ\x90\x00\x03\x00\x00\x00\x04\x00" * 50, b"just plain text, no tags"):
        with pytest.raises(UploadError):
            validate_content("x.html", bad)


# ----------------------------------------------------------------------- JavaScript-rendered pages
SPA = """<html><head><title>Bid 2024-17 | Portal</title>
<meta name="description" content="Library furniture replacement: 40 study carrels and 120 reading chairs.">
<script src="/static/app.js"></script>
<script type="application/ld+json">{"@context": "https://schema.org", "@type": "Event",
 "name": "Pre-bid walkthrough at Cafe Hall", "startDate": "2024-06-03", "url": "https://x.example/e"}</script>
<script id="__NEXT_DATA__" type="application/json">{"props": {"pageProps": {"bid": {
 "bidNumber": "LIB-2024-17", "closingDate": "June 14, 2024 2:00 PM", "id": "a1b2c3d4e5",
 "summary": "<p>Vendors must provide a <b>5% bid bond</b>.</p>", "className": "card big"}}}}</script>
</head><body><noscript>Please enable JavaScript to use this site.</noscript><div id="root"></div></body></html>"""


def test_scripted_page_indexes_embedded_text():
    blocks, title, warnings = HtmlParser().parse_html_with_info(SPA)
    texts = [b.text for b in blocks]
    joined = "\n".join(texts)
    assert texts[0] == "Bid 2024-17 | Portal"  # title kept as heading
    assert "Library furniture replacement: 40 study carrels and 120 reading chairs." in texts
    assert "Bid number: LIB-2024-17" in texts and "Closing date: June 14, 2024 2:00 PM" in texts
    assert "Summary: Vendors must provide a 5% bid bond ." in texts or "5% bid bond" in joined
    assert "Name: Pre-bid walkthrough at Cafe Hall" in texts
    for junk in ("a1b2c3d4e5", "card big", "https://", "enable JavaScript", "schema.org"):
        assert junk not in joined
    assert warnings and "JavaScript" in warnings[0]


def test_empty_scripted_page_fails_with_actionable_message(engine, tmp_path):
    f = tmp_path / "app.html"
    f.write_text("<html><head><title>App</title><script src='a.js'></script></head>"
                 "<body><div id='root'></div></body></html>", encoding="utf-8")
    status = engine.indexing.index_file(f, "SpaBid")
    assert status.status == "failed"
    assert "No extractable text" in status.message and "Save as" in status.message


def test_normal_page_ignores_fallback_sources():
    html = BID_HTML.replace("<head>", '<head><meta name="description" content="Portal marketing blurb here.">')
    blocks, _, warnings = HtmlParser().parse_html_with_info(html)
    assert "Portal marketing blurb here." not in [b.text for b in blocks] and warnings == []


# ----------------------------------------------------------------------- doc types
@pytest.mark.parametrize("name,head,expected", [
    ("Addendum 2 - Bid 555.html", "", DocType.ADDENDUM),
    ("notice.htm", "Addendum No. 3 to RFQ-555 The closing date is extended", DocType.ADDENDUM),
    ("Widget Specs.html", "", DocType.SPECS),
    ("Vendor Affidavit.htm", "", DocType.AFFIDAVIT),
    ("RFQ-555 Bid Information.html", "Solicitation RFQ-555 Bid Information closing date", DocType.BID_PAGE),
    ("Bid Information.html", "", DocType.BID_PAGE),
])
def test_html_doc_types(name, head, expected):
    assert classify_doc_type(name, head) == expected


def test_html_addendum_indexed_with_number_and_filterable(engine, tmp_path):
    f = tmp_path / "Addendum 2 - Bid 555.html"
    f.write_text("<html><body><main><h1>Addendum No. 2</h1><p>The bid opening is moved to May 9, 2024.</p>"
                 "</main></body></html>", encoding="utf-8")
    status = engine.indexing.index_file(f, "AddBid")
    assert status.status == "indexed" and status.doc_type == "addendum"
    assert extract_addendum_number(f.name) == 2
    res = engine.retrieval.search("bid opening moved", mode="bm25",
                                  filters=SearchFilters(bid_id="AddBid", doc_type="addendum", addendum_number=2))
    assert res and res[0].file_name == f.name and res[0].addendum_number == 2


# ----------------------------------------------------------------------- citations
def test_pageless_citation_names_the_section():
    base = dict(file_name="page.html", bid_id="B", doc_type="bid_page", chunk_id="c", source_path="/p")
    assert Citation(**base).label == "page.html"
    assert Citation(**base, section="Bid Information > Dates").label == 'page.html, section "Dates"'
    long = Citation(**base, section="x" * 80).label
    assert long.endswith('..."') and len(long) < 90
    assert Citation(**{**base, "file_name": "a.pdf"}, page_number=3, section="Dates").label == "a.pdf, p. 3"


def test_html_search_result_citation_has_section(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    res = engine.retrieval.search("closing date", filters=SearchFilters(doc_type="bid_page"), mode="bm25")
    hit = next(r for r in res if r.section)
    assert hit.citation.startswith(f'{hit.file_name}, section "') and "p. " not in hit.citation
