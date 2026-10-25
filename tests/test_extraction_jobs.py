"""UI / API extraction jobs (services.extraction_jobs, /extract/*) - offline: fake embedder,
stub LLMs, outputs in a temp folder (conftest sets OUTPUT_DIR), no OpenRouter call.

* bid selection and isolation: two bids indexed side by side, only the selected one reaches
  the model and the citations;
* the four output files, named after the bid, matching the existing BidExtraction schema;
* addenda included and reconciled (Due Date changed by Addendum 1);
* backups before overwrite (byte-identical, never into an existing folder, failure aborts
  before any model call) and the overwrite confirmation;
* rate-limit / auth / payment errors stop the run's requests; cancel between model calls.
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

from agents import BidNotIndexedError
from api import routes
from extraction.fields import FIELD_NAMES
from extraction.schemas import BidExtraction
from jobcontrol import JobCancelled, JobControl, set_current
from services import extraction_jobs as xj
from services.llm_service import LLMConfigError, LLMRateLimitError, ModelNotAllowedError, OpenRouterLLM
from services.uploads import UploadError
from tests.conftest import make_pdf
from tests.test_extraction import ANSWERS, ORIGINAL, NeedleLLM

KW = {"per_query": 10, "mode": "hybrid"}  # same retrieval settings as the agent parity tests
OTHER_BID_MARKERS = ("RFQ-999", "Gadget", "May 5, 2025")


@pytest.fixture
def two_bids(engine, bid_folder, settings):
    other = settings.bids_root / "BidB"
    other.mkdir()
    make_pdf(other / "RFQ-999 Gadget Supply.pdf", [
        "Request for Quote RFQ-999 Gadget Supply\nProposals are due May 5, 2025 at 3:00 PM.\n"
        "A bid bond of 10% is required. Gadgets require a five year warranty."])
    make_pdf(other / "Addendum 2 RFQ-999.pdf", ["ADDENDUM No. 2\nThe due date is extended to May 20, 2025."])
    engine.indexing.index_folder(bid_folder)
    engine.indexing.index_folder(other)
    engine.llm = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    return engine


def run(engine, **kw):
    return xj.run_bid_extraction(engine, "BidA", **{**KW, **kw})


# ----------------------------------------------------------------------- preflight / selection
def test_preflight_lists_only_the_selected_bid_and_its_addenda(two_bids, settings):
    pre = xj.preflight(two_bids, "BidA")
    assert pre["indexed"] and {d["file_name"] for d in pre["documents"]} == {
        "RFP-555 Widget Supply.pdf", "Addendum 1 RFP-555.pdf", "Widget_Specs.pdf", "Vendor_Affidavit.pdf",
        "RFQ-555 Bid Information.html"}
    assert pre["addenda"] == [{"file_name": "Addendum 1 RFP-555.pdf", "addendum_number": 1,
                               "document_date": pre["addenda"][0]["document_date"]}]
    assert {k: o["path"] for k, o in pre["outputs"].items()} == {
        "json": str(settings.output_dir / "BidA.json"), "report": str(settings.output_dir / "BidA.report.md"),
        "trace": str(settings.output_dir / "BidA.trace.jsonl"), "cli_log": str(settings.output_dir / "BidA.cli.log")}
    assert pre["existing"] == [] and pre["default_model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert two_bids.llm.calls == []  # preflight never calls the model


def test_unsafe_or_unknown_bid_ids_are_rejected_before_any_call(two_bids, settings):
    for bad in ("../evil", "a/b", ""):
        with pytest.raises(UploadError):
            xj.output_paths(bad, settings.output_dir)
    with pytest.raises(BidNotIndexedError):
        xj.run_bid_extraction(two_bids, "BidZ", **KW)
    assert two_bids.llm.calls == [] and not settings.output_dir.exists()


# ----------------------------------------------------------------------- outputs, schema, isolation
def test_run_writes_the_four_outputs_with_the_existing_schema(two_bids, settings):
    res = run(two_bids)
    out = settings.output_dir
    assert res.outputs == {k: str(out / f"BidA{s}") for k, s in xj.OUTPUT_KINDS.items()}
    rec = BidExtraction.model_validate_json((out / "BidA.json").read_text(encoding="utf-8"))
    assert list(rec.fields) == list(FIELD_NAMES) and rec.bid_id == "BidA"
    assert rec.fields["Bid Number"].value == "RFQ-555" and rec.fields["Bid Number"].sources
    for f in rec.fields.values():  # every field carries the schema's keys
        assert f.status in ("verified", "unverified", "not_found", "error") and f.attempts >= 1
        assert f.value is None or f.sources or f.status != "verified"
    assert rec.validation.passed == res.validation["passed"] and rec.llm_calls == res.llm_calls > 0

    events = [json.loads(line) for line in (out / "BidA.trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "done" and {e.get("run_id") for e in events if e.get("run_id")} == {res.run_id}
    log = (out / "BidA.cli.log").read_text(encoding="utf-8")
    assert res.run_id in log and "passed=" in log and str(out / "BidA.json") in log
    assert "Addendum 1 RFP-555.pdf" in log  # documents used are listed
    assert (out / "BidA.report.md").read_text(encoding="utf-8").startswith("# ")


def test_only_the_selected_bid_reaches_the_model_and_the_citations(two_bids, settings):
    run(two_bids)
    sent = "\n".join(t for c in two_bids.llm.calls for t in c["passages"].values())
    assert sent and not any(m in sent for m in OTHER_BID_MARKERS)
    rec = BidExtraction.model_validate_json((settings.output_dir / "BidA.json").read_text(encoding="utf-8"))
    bida_files = {d["file_name"] for d in two_bids.retrieval.documents("BidA")}
    cited = {s.file for f in rec.fields.values() for s in f.sources}
    cited |= {s.file for c in rec.addendum_changes for s in c.sources + c.previous_sources}
    assert cited and cited <= bida_files
    for name in ("BidA.json", "BidA.trace.jsonl"):
        assert not any(m in (settings.output_dir / name).read_text(encoding="utf-8") for m in OTHER_BID_MARKERS)


def test_addendum_is_used_and_reconciled(two_bids, settings):
    res = run(two_bids)
    assert [c["field"] for c in res.addendum_changes] == ["Due Date"]
    rec = BidExtraction.model_validate_json((settings.output_dir / "BidA.json").read_text(encoding="utf-8"))
    change = rec.addendum_changes[0]
    assert "April 15, 2024" in change.value and "April 1, 2024" in change.previous_value
    assert change.addendum_number == 1 and change.sources[0].file == "Addendum 1 RFP-555.pdf"


# ----------------------------------------------------------------------- overwrite and backups
def test_overwrite_needs_confirmation_and_previous_files_are_backed_up(two_bids, settings):
    run(two_bids)
    out = settings.output_dir
    (out / "BidA.json").write_text('{"marker": "previous run"}', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in out.glob("BidA.*")}
    keep = out / "backup_20200101T000000Z_BidA"  # an older backup: must stay untouched
    keep.mkdir()
    (keep / "BidA.json").write_text("old backup", encoding="utf-8")
    calls = len(two_bids.llm.calls)

    with pytest.raises(xj.OverwriteNotConfirmed):
        run(two_bids)
    assert len(two_bids.llm.calls) == calls and {p.name: p.read_bytes() for p in out.glob("BidA.*")} == before

    res = run(two_bids, confirm_overwrite=True)
    backup = out / res.backup_dir.split("\\")[-1].split("/")[-1]
    assert backup.is_dir() and backup != keep
    assert {p.name: p.read_bytes() for p in backup.iterdir()} == before  # byte-identical copies
    assert "previous run" not in (out / "BidA.json").read_text(encoding="utf-8")  # new output written
    assert (keep / "BidA.json").read_text(encoding="utf-8") == "old backup"


def test_backup_failure_aborts_before_any_model_call(two_bids, settings, monkeypatch):
    out = settings.output_dir
    out.mkdir(parents=True)
    (out / "BidA.json").write_text("previous", encoding="utf-8")

    def broken_copy(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(xj.shutil, "copy2", broken_copy)
    with pytest.raises(xj.BackupError, match="disk full"):
        run(two_bids, confirm_overwrite=True)
    assert two_bids.llm.calls == [] and (out / "BidA.json").read_text(encoding="utf-8") == "previous"


def test_an_existing_backup_folder_is_never_reused(two_bids, settings, monkeypatch):
    out = settings.output_dir
    out.mkdir(parents=True)
    (out / "BidA.json").write_text("previous", encoding="utf-8")
    monkeypatch.setattr(xj, "_utc_stamp", lambda: "20260101T000000Z")
    taken = out / "backup_20260101T000000Z_BidA"
    taken.mkdir()
    (taken / "BidA.json").write_text("earlier backup", encoding="utf-8")
    with pytest.raises(xj.BackupError, match="already exists"):
        run(two_bids, confirm_overwrite=True)
    assert (taken / "BidA.json").read_text(encoding="utf-8") == "earlier backup" and two_bids.llm.calls == []


def test_backup_copy_mismatch_is_detected(settings, tmp_path, monkeypatch):
    src = tmp_path / "BidA.json"
    src.write_text("original", encoding="utf-8")
    monkeypatch.setattr(xj.shutil, "copy2", lambda a, b: open(b, "w").write("corrupted"))
    with pytest.raises(xj.BackupError, match="not identical"):
        xj.backup_outputs({"json": src}, tmp_path, "BidA")


# ----------------------------------------------------------------------- stop errors and cancel
class FailingLLM:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def complete(self, messages, model=None):
        self.calls += 1
        raise self.exc


@pytest.mark.parametrize("exc", [LLMRateLimitError("rate limited: free-models-per-day"),
                                 LLMConfigError("OpenRouter rejected the API key (401)"),
                                 ModelNotAllowedError("OpenRouter requires payment (402)")])
def test_quota_auth_or_payment_error_stops_all_further_requests(two_bids, settings, exc):
    two_bids.llm = FailingLLM(exc)
    with pytest.raises(xj.ExtractionStopped) as info:
        run(two_bids)
    assert two_bids.llm.calls == 1  # every later call was refused without being sent
    assert "not replaced" in str(info.value)
    out = settings.output_dir
    assert not (out / "BidA.json").exists()  # main outputs untouched
    partial = [p for p in out.iterdir() if p.name.startswith("partial_")]
    assert len(partial) == 1 and {p.name for p in partial[0].iterdir()} == {
        "BidA.json", "BidA.report.md", "BidA.trace.jsonl", "BidA.cli.log"}
    rec = BidExtraction.model_validate_json((partial[0] / "BidA.json").read_text(encoding="utf-8"))
    assert all(f.status in ("error", "not_found") for f in rec.fields.values())
    assert "STOPPED" in (partial[0] / "BidA.cli.log").read_text(encoding="utf-8")


def test_cancel_stops_before_the_next_model_call_and_writes_nothing(two_bids, settings):
    control = JobControl()
    inner = two_bids.llm

    class CancelAfterFirst:
        calls = 0

        def complete(self, messages, model=None):
            self.calls += 1
            control.cancel_event.set()  # the user presses Cancel while the first call runs
            return inner.complete(messages, model)

    two_bids.llm = llm = CancelAfterFirst()
    set_current(control)
    try:
        with pytest.raises(JobCancelled, match="no output files written"):
            run(two_bids)
    finally:
        set_current(None)
    assert llm.calls == 1 and not settings.output_dir.exists()


def test_guard_passes_through_and_counts():
    llm = NeedleLLM({})
    guard = xj.StopGuardLLM(llm)
    guard.llm = llm
    with pytest.raises(Exception):
        guard.complete([{"role": "system", "content": ""}, {"role": "user", "content": "no keys"}])
    assert guard.sent == 1 and guard.stopped is None  # a non-stop error does not stop the run


# ----------------------------------------------------------------------- API + job queue
@pytest.fixture
def client(two_bids):
    routes.app.dependency_overrides[routes.engine] = lambda: two_bids
    yield TestClient(routes.app)
    routes.app.dependency_overrides.clear()


def wait(client, job_id):
    for _ in range(300):
        job = client.get(f"/index/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def test_extract_job_endpoint_runs_in_the_job_queue(client, settings):
    pre = client.get("/extract/preflight", params={"bid_id": "BidA"}).json()
    assert pre["indexed"] and pre["existing"] == []
    r = client.post("/extract/jobs", json={"bid_id": "BidA"})
    assert r.status_code == 202 and r.json()["kind"] == "extract"
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in r.json()["description"]  # default model requested
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    rep = job["report"]
    assert rep["bid_id"] == "BidA" and rep["model"] == "stub/extractor:free"  # the model that answered
    assert rep["outputs"]["json"] == str(settings.output_dir / "BidA.json") and len(rep["fields"]) == 20
    assert job["progress"]["done"] == job["progress"]["total"] > 0
    # outputs exist now: a second run needs confirmation, then backs up
    assert client.post("/extract/jobs", json={"bid_id": "BidA"}).status_code == 409
    r = client.post("/extract/jobs", json={"bid_id": "BidA", "confirm_overwrite": True})
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded" and job["report"]["backup_dir"]


def test_extract_job_rejections(client, two_bids):
    assert client.get("/extract/preflight", params={"bid_id": "../evil"}).status_code == 400
    assert client.post("/extract/jobs", json={"bid_id": "BidZ"}).status_code == 404
    assert client.post("/extract/jobs", json={"bid_id": "BidA", "model": "openai/gpt-paid"}).status_code == 400
    two_bids.llm = OpenRouterLLM.__new__(OpenRouterLLM)  # the real client type, never called
    assert client.post("/extract/jobs", json={"bid_id": "BidA"}).status_code == 400  # no API key
    assert client.get("/index/jobs").json()["jobs"] == [] or all(
        j["kind"] in ("index", "extract") for j in client.get("/index/jobs").json()["jobs"])


def test_failed_extract_job_reports_the_stop_reason(client, two_bids):
    two_bids.llm = FailingLLM(LLMRateLimitError("rate limited: free-models-per-day"))
    job = wait(client, client.post("/extract/jobs", json={"bid_id": "BidA"}).json()["job_id"])
    assert job["status"] == "failed" and "ExtractionStopped" in job["error"]
    assert "no further model requests" in job["error"].lower() and two_bids.llm.calls == 1


def test_index_jobs_still_report_as_before(client, settings):
    r = client.post("/index/jobs", json={"bid_id": "BidA"})
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "succeeded" and job["kind"] == "index"
    assert {f["status"] for f in job["report"]["files"]} == {"skipped_unchanged"}
