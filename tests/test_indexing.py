from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.pdf_parser import PdfParser
from services.container import build_engine
from tests.conftest import FakeEmbedder, make_pdf


def _statuses(report):
    return {f.file_name: f.status for f in report.files}


def test_index_folder_indexes_all_supported_files(engine, bid_folder):
    (bid_folder / "notes.txt").write_text("ignored")
    report = engine.indexing.index_folder(bid_folder)
    st = _statuses(report)
    assert len(st) == 5 and set(st.values()) == {"indexed"}
    assert "notes.txt" not in st
    assert engine.vector_store.count() == report.total_chunks_added > 0
    types = {f.file_name: f.doc_type for f in report.files}
    assert types["Addendum 1 RFP-555.pdf"] == "addendum"
    assert types["Widget_Specs.pdf"] == "specs"
    assert types["Vendor_Affidavit.pdf"] == "affidavit"
    assert types["RFQ-555 Bid Information.html"] == "bid_page"
    assert types["RFP-555 Widget Supply.pdf"] == "rfp"


def test_reindex_skips_unchanged(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    count = engine.vector_store.count()
    report = engine.indexing.index_folder(bid_folder)
    assert set(_statuses(report).values()) == {"skipped_unchanged"}
    assert engine.vector_store.count() == count


def test_changed_file_replaces_old_chunks(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    target = bid_folder / "Addendum 1 RFP-555.pdf"
    old_doc_id = engine.manifest.get(target)["doc_id"]
    make_pdf(target, ["ADDENDUM No. 1\nThe due date is now May 30, 2024 at 3:00 PM."])
    report = engine.indexing.index_folder(bid_folder)
    st = _statuses(report)
    assert st["Addendum 1 RFP-555.pdf"] == "indexed"
    assert sum(v == "skipped_unchanged" for v in st.values()) == 4
    assert engine.vector_store.doc_chunk_count(old_doc_id) == 0
    texts = " ".join(c.text for c in engine.vector_store.all_chunks())
    assert "May 30, 2024" in texts and "April 15, 2024" not in texts


def test_same_content_twice_is_not_duplicated(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    count = engine.vector_store.count()
    copy = bid_folder / "copy of addendum.pdf"
    copy.write_bytes((bid_folder / "Addendum 1 RFP-555.pdf").read_bytes())
    status = engine.indexing.index_file(copy, "BidA")
    assert status.status == "skipped_duplicate"
    assert engine.vector_store.count() == count


def test_force_reindex_does_not_duplicate_chunks(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    count = engine.vector_store.count()
    engine.indexing.index_folder(bid_folder, force=True)
    assert engine.vector_store.count() == count  # deterministic ids -> upsert


def test_deleted_file_is_pruned(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    victim = bid_folder / "Vendor_Affidavit.pdf"
    doc_id = engine.manifest.get(victim)["doc_id"]
    victim.unlink()
    report = engine.indexing.index_folder(bid_folder)
    assert _statuses(report)["Vendor_Affidavit.pdf"] == "removed"
    assert engine.vector_store.doc_chunk_count(doc_id) == 0
    assert engine.manifest.get(victim) is None


def test_unreadable_and_empty_files_fail_gracefully(engine, bid_folder):
    (bid_folder / "corrupt.pdf").write_bytes(b"%PDF-1.4 garbage")
    make_pdf(bid_folder / "scanned_blank.pdf", ["", ""])
    report = engine.indexing.index_folder(bid_folder)
    st = _statuses(report)
    assert st["corrupt.pdf"] == "failed"
    assert st["scanned_blank.pdf"] == "failed"
    assert sum(v == "indexed" for v in st.values()) == 5  # others unaffected
    failed = next(f for f in report.files if f.file_name == "scanned_blank.pdf")
    assert "No extractable text" in failed.message
    assert engine.manifest.get(bid_folder / "corrupt.pdf")["status"] == "failed"


def test_index_persists_across_restart(settings, bid_folder):
    def make():
        return build_engine(settings, embedder=FakeEmbedder(), loader=DocumentLoader(PdfParser(backend="pymupdf")))

    e1 = make()
    e1.indexing.index_folder(bid_folder)
    count = e1.vector_store.count()
    e2 = make()  # new process-equivalent: re-opens Chroma + manifest from disk
    assert e2.vector_store.count() == count
    report = e2.indexing.index_folder(bid_folder)
    assert set(_statuses(report).values()) == {"skipped_unchanged"}
    assert e2.retrieval.search("bid bond", filters=None, mode="bm25")


def test_new_bid_folder_is_discovered_without_code_changes(engine, settings, bid_folder):
    new = settings.bids_root / "Bid9"
    new.mkdir()
    make_pdf(new / "Invitation to Bid 77.pdf", ["Invitation for Bids 77. Bids due 01/02/2025."])
    report = engine.indexing.index_all()
    assert {f.bid_id for f in report.files} == {"BidA", "Bid9"}
    assert {b["bid_id"] for b in engine.retrieval.bids()} == {"BidA", "Bid9"}


def test_index_all_includes_upload_folders(engine, settings, bid_folder):
    up = settings.uploads_dir / "BidA"
    up.mkdir(parents=True)
    make_pdf(up / "Addendum 2.pdf", ["ADDENDUM No. 2\nPre-bid meeting cancelled."])
    report = engine.indexing.index_all()
    added = next(f for f in report.files if f.file_name == "Addendum 2.pdf")
    assert added.status == "indexed" and added.bid_id == "BidA"
    assert Path(added.source_path).parent == up.resolve()
