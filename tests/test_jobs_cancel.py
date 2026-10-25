"""Cancellation of background indexing jobs, OCR override and large-PDF profiling."""
import threading
import time
from pathlib import Path

import pytest

from jobcontrol import JobCancelled, JobControl, check_cancelled, report_detail, set_current
from schemas import IndexReport
from services.jobs import JobManager
from services.uploads import inspect_pdf
from tests.conftest import make_pdf


def _wait(jm, job_id, statuses=("succeeded", "failed", "cancelled"), timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        job = jm.get(job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job stuck: {jm.get(job_id)}")


# ----------------------------------------------------------------------- JobManager
def test_cancel_running_job_stops_at_safe_point():
    jm = JobManager()
    started, steps = threading.Event(), []

    def work(progress):
        started.set()
        for i in range(500):
            check_cancelled()
            report_detail(f"step {i}")
            steps.append(i)
            time.sleep(0.01)
        return IndexReport()

    job = jm.submit("long", work)
    started.wait(5)
    time.sleep(0.05)
    assert jm.cancel(job["job_id"])["status"] == "cancelling"
    done = _wait(jm, job["job_id"])
    assert done["status"] == "cancelled" and "Cancelled by user" in done["error"]
    assert len(steps) < 500 and done["progress"]["detail"].startswith("step ")
    assert done["finished_at"] is not None


def test_cancel_queued_job_never_runs():
    jm = JobManager()
    release, ran = threading.Event(), []

    def blocker(progress):
        release.wait(5)
        return IndexReport()

    def second(progress):
        ran.append(True)
        return IndexReport()

    j1 = jm.submit("first", blocker)
    j2 = jm.submit("second", second)
    assert jm.queue_position(j2["job_id"]) == 1
    assert jm.cancel(j2["job_id"])["status"] == "cancelled"
    release.set()
    assert _wait(jm, j1["job_id"])["status"] == "succeeded"
    time.sleep(0.1)
    assert ran == [] and jm.get(j2["job_id"])["status"] == "cancelled"


def test_cancel_unknown_and_finished_jobs():
    jm = JobManager()
    assert jm.cancel("nope") is None
    j = jm.submit("quick", lambda p: IndexReport())
    _wait(jm, j["job_id"])
    assert jm.cancel(j["job_id"])["status"] == "succeeded"  # finished jobs are unaffected


def test_job_control_is_noop_outside_jobs():
    set_current(None)
    check_cancelled()  # must not raise
    report_detail("ignored")


# ----------------------------------------------------------------------- indexing
@pytest.fixture
def ctrl():
    c = JobControl()
    set_current(c)
    yield c
    set_current(None)


def test_indexing_cancelled_between_files(engine, bid_folder, ctrl):
    files = sorted(bid_folder.glob("*.pdf"))[:3]

    def progress(done, total, name):
        if done == 1:  # first file finished -> user clicks Cancel
            ctrl.cancel_event.set()

    with pytest.raises(JobCancelled):
        engine.indexing.index_files(files, "BidA", progress=progress)
    assert engine.manifest.get(files[0])["status"] == "indexed"  # finished work is kept
    assert engine.manifest.get(files[1]) is None  # cancelled file: no failure record
    assert engine.manifest.get(files[2]) is None


def test_cancelled_reindex_keeps_previous_version(engine, bid_folder, ctrl):
    target = bid_folder / "Addendum 1 RFP-555.pdf"
    engine.indexing.index_file(target, "BidA")
    old = engine.manifest.get(target)
    make_pdf(target, ["ADDENDUM No. 1\nCompletely new content about delivery."])
    ctrl.cancel_event.set()
    with pytest.raises(JobCancelled):
        engine.indexing.index_file(target, "BidA")
    assert engine.vector_store.doc_chunk_count(old["doc_id"]) > 0  # old chunks not deleted
    assert engine.manifest.get(target)["content_hash"] == old["content_hash"]


def test_ocr_override_reaches_parser_and_is_recorded(engine, bid_folder):
    seen = []
    pdf = engine.indexing.loader.pdf
    orig = pdf.parse

    def spy(path, ocr=None):
        seen.append(ocr)
        return orig(path, ocr=ocr)

    pdf.parse = spy
    target = bid_folder / "Widget_Specs.pdf"
    engine.indexing.index_files([target], "BidA", ocr=False)
    assert seen == [False]
    assert engine.manifest.get(target)["ocr_used"] is False
    engine.indexing.index_files([target], "BidA", force=True)
    assert seen[-1] is None  # default: configured setting
    assert engine.manifest.get(target)["ocr_used"] == pdf.ocr


# ----------------------------------------------------------------------- profiling
def test_inspect_pdf_flags_large_documents(settings, tmp_path):
    text = "This page has a normal text layer with enough characters."
    small = make_pdf(tmp_path / "small.pdf", [text, text])
    big = make_pdf(tmp_path / "big.pdf", [text] * 6 + ["", ""])
    settings.large_pdf_pages = 5
    p_small, p_big = inspect_pdf(small), inspect_pdf(big)
    assert p_small == {"pages": 2, "images": 0, "pages_without_text": 0,
                       "estimated_seconds_with_ocr": round(2 * settings.ocr_seconds_per_page), "large": False}
    assert p_big["pages"] == 8 and p_big["pages_without_text"] == 2 and p_big["large"] is True
    assert p_big["estimated_seconds_with_ocr"] == round(8 * settings.ocr_seconds_per_page)


# ----------------------------------------------------------------------- Docling batches
@pytest.mark.slow
def test_docling_page_batches_report_progress_and_cancel(settings):
    pytest.importorskip("docling")
    from config import PROJECT_ROOT
    from ingestion.pdf_parser import PdfParser

    pdf = next(iter(sorted(PROJECT_ROOT.glob("Bid*/Contract_Affidavit.pdf"))), None)
    models = PROJECT_ROOT / "data" / "models" / "docling"
    if pdf is None or not models.exists():
        pytest.skip("assignment PDF or Docling models not available")
    settings.docling_artifacts_dir = models
    settings.pdf_batch_pages = 1
    parser = PdfParser(backend="docling", ocr=False)

    details = []
    set_current(JobControl(on_detail=details.append))
    try:
        res = parser.parse(Path(pdf))
    finally:
        set_current(None)
    assert res.parser == "docling" and res.num_pages == 3
    assert {b.page for b in res.blocks} == {1, 2, 3}  # page numbers preserved across batches
    assert details == ["Contract_Affidavit.pdf: pages 1/3", "Contract_Affidavit.pdf: pages 2/3",
                       "Contract_Affidavit.pdf: pages 3/3"]

    c = JobControl()
    c.on_detail = lambda d: c.cancel_event.set()  # cancel right after the first batch
    set_current(c)
    try:
        with pytest.raises(JobCancelled):  # not swallowed by the PyMuPDF fallback
            parser.parse(Path(pdf))
    finally:
        set_current(None)
