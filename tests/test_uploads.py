import pytest

from services.uploads import UploadError, safe_filename, store_upload, validate_bid_id, validate_content
from tests.conftest import BID_HTML, make_pdf


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\Windows\\evil.html", "evil.html"),
        ("my file (v2).PDF", "my file (v2).pdf"),
        ("spec<>:\"|?*.pdf", "spec_.pdf"),
        ("CON.pdf", "_CON.pdf"),
    ],
)
def test_safe_filename(settings, raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["virus.exe", "noext", ".pdf", "archive.pdf.zip"])
def test_safe_filename_rejects(settings, raw):
    with pytest.raises(UploadError):
        safe_filename(raw)


@pytest.mark.parametrize("bid", ["../Bid1", "Bid 1", "", "a" * 65, "-lead"])
def test_invalid_bid_ids(bid):
    with pytest.raises(UploadError):
        validate_bid_id(bid)


def test_content_validation(settings, tmp_path):
    validate_content("a.html", BID_HTML.encode())
    pdf = make_pdf(tmp_path / "x.pdf", ["hello"]).read_bytes()
    validate_content("x.pdf", pdf)
    with pytest.raises(UploadError, match="not a valid PDF"):
        validate_content("fake.pdf", b"MZ\x90\x00 this is an exe")
    with pytest.raises(UploadError, match="HTML"):
        validate_content("fake.html", b"\x00\x01binary")
    with pytest.raises(UploadError, match="empty"):
        validate_content("e.pdf", b"")
    with pytest.raises(UploadError, match="limit"):
        validate_content("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024))  # MAX_UPLOAD_MB=1


def test_store_upload_dedup_and_name_collision(settings, tmp_path):
    a = make_pdf(tmp_path / "a.pdf", ["first"]).read_bytes()
    b = make_pdf(tmp_path / "b.pdf", ["second"]).read_bytes()
    p1, existed1 = store_upload("Bid7", "../Addendum.pdf", a)
    p2, existed2 = store_upload("Bid7", "Addendum.pdf", a)
    p3, existed3 = store_upload("Bid7", "Addendum.pdf", b)
    assert (existed1, existed2, existed3) == (False, True, False)
    assert p1 == p2 and p1.parent == (settings.uploads_dir / "Bid7").resolve()
    assert p3 != p1 and p3.name.startswith("Addendum_")
    assert all(settings.uploads_dir.resolve() in p.parents for p in (p1, p3))


def test_upload_then_index_into_existing_bid(engine, settings, bid_folder):
    engine.indexing.index_folder(bid_folder)
    data = make_pdf(settings.data_dir / "tmp_upload.pdf",
                    ["ADDENDUM No. 2\nThe pre-bid meeting is cancelled."]).read_bytes()
    path, _ = store_upload("BidA", "Addendum 2 RFP-555.pdf", data)
    report = engine.indexing.index_files([path], "BidA")
    assert report.files[0].status == "indexed" and report.files[0].doc_type == "addendum"
    # same file uploaded again -> no new chunks
    count = engine.vector_store.count()
    path2, existed = store_upload("BidA", "Addendum 2 RFP-555.pdf", data)
    assert existed and path2 == path
    assert engine.indexing.index_files([path2], "BidA").files[0].status == "skipped_unchanged"
    assert engine.vector_store.count() == count
    hits = engine.retrieval.search("pre-bid meeting cancelled", mode="bm25")
    assert hits[0].bid_id == "BidA" and hits[0].addendum_number == 2


def test_upload_creates_new_bid(engine, settings):
    data = BID_HTML.encode()
    path, _ = store_upload("NewBid", "portal page.html", data)
    report = engine.indexing.index_files([path], "NewBid")
    assert report.files[0].status == "indexed"
    assert "NewBid" in {b["bid_id"] for b in engine.retrieval.bids()}
