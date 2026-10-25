"""FastAPI endpoint tests (in-process TestClient, fake embedder, stubbed LLM)."""
import pytest
from fastapi.testclient import TestClient

from api import routes
from tests.conftest import BID_HTML, make_pdf
from tests.test_qa_citations import StubLLM


@pytest.fixture
def client(engine, bid_folder):
    engine.qa.llm = StubLLM("Proposals are due April 15, 2024 per Addendum 1 [1].")
    routes.app.dependency_overrides[routes.engine] = lambda: engine
    yield TestClient(routes.app)
    routes.app.dependency_overrides.clear()


def test_index_search_ask_flow(client, bid_folder):
    r = client.post("/index", json={"bid_path": str(bid_folder)})
    assert r.status_code == 200
    assert {f["status"] for f in r.json()["files"]} == {"indexed"}

    r = client.get("/search", params={"q": "due date", "bid_id": "BidA", "doc_type": "addendum", "top_k": 3, "mode": "hybrid"})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res and all(x["doc_type"] == "addendum" for x in res)
    for key in ("text", "file_name", "page_number", "bid_id", "doc_type", "chunk_id", "score", "rank", "citation"):
        assert key in res[0]

    r = client.post("/ask", json={"question": "When are proposals due?", "bid_id": "BidA", "mode": "bm25"})
    assert r.status_code == 200
    ans = r.json()
    assert ans["found"] and ans["citations"][0]["file_name"].endswith(".pdf")

    assert client.get("/health").json()["documents"] == 5
    assert client.get("/bids").json()["bids"][0]["bid_id"] == "BidA"


def test_background_index_job(client, bid_folder):
    import time

    r = client.post("/index/jobs", json={"bid_path": str(bid_folder)})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/index/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "succeeded", job
    assert {f["status"] for f in job["report"]["files"]} == {"indexed"}
    assert job["progress"]["done"] == job["progress"]["total"] == 5
    assert any(j["job_id"] == job_id for j in client.get("/index/jobs").json()["jobs"])
    # invalid requests are rejected before a job is created
    assert client.post("/index/jobs", json={"bid_id": "NoSuchBid"}).status_code == 404
    assert client.post("/index/jobs", json={}).status_code == 400
    assert client.get("/index/jobs/doesnotexist").status_code == 404


def test_upload_profile_ocr_override_and_cancel_endpoint(client, engine, settings, tmp_path):
    import time

    settings.large_pdf_pages = 1
    pdf = make_pdf(tmp_path / "big.pdf", ["Page one has a normal text layer.", "Page two has a normal text layer.",
                                          ""]).read_bytes()
    up = client.post("/upload", data={"bid_id": "BigBid", "index": "false"},
                     files=[("files", ("big.pdf", pdf, "application/pdf"))]).json()
    prof = up["saved"][0]["pdf"]
    assert prof["pages"] == 3 and prof["pages_without_text"] == 1 and prof["large"] is True
    assert up["report"] is None  # not indexed: client decides first

    r = client.post("/index/jobs", json={"bid_id": "BigBid", "paths": [up["saved"][0]["path"]], "ocr": False})
    assert r.status_code == 202 and r.json()["description"].endswith("(OCR off)")
    job_id = r.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/index/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded" and job["queue_position"] == 0
    assert engine.manifest.get(up["saved"][0]["path"])["ocr_used"] is False
    # cancelling a finished job changes nothing; unknown job -> 404
    assert client.post(f"/index/jobs/{job_id}/cancel").json()["status"] == "succeeded"
    assert client.post("/index/jobs/doesnotexist/cancel").status_code == 404


def test_search_validation(client):
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "mode": "bogus"}).status_code == 400
    assert client.get("/search", params={"q": "x", "top_k": 500}).status_code == 422


def test_index_path_traversal_blocked(client, tmp_path):
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    r = client.post("/index", json={"bid_path": str(outside)})
    assert r.status_code == 400
    assert client.post("/index", json={}).status_code == 400


def test_upload_endpoint(client, settings, tmp_path):
    pdf = make_pdf(tmp_path / "add.pdf", ["ADDENDUM No. 2\nBid opening moved online."]).read_bytes()
    files = [
        ("files", ("Addendum 2.pdf", pdf, "application/pdf")),
        ("files", ("page.html", BID_HTML.encode(), "text/html")),
        ("files", ("malware.exe", b"MZ...", "application/octet-stream")),
        ("files", ("fake.pdf", b"not a pdf", "application/pdf")),
    ]
    r = client.post("/upload", data={"bid_id": "Bid3", "index": "true"}, files=files)
    assert r.status_code == 200
    body = r.json()
    assert {s["stored_as"] for s in body["saved"]} == {"Addendum 2.pdf", "page.html"}
    assert {x["original_name"] for x in body["rejected"]} == {"malware.exe", "fake.pdf"}
    assert {f["status"] for f in body["report"]["files"]} == {"indexed"}
    assert (settings.uploads_dir / "Bid3" / "Addendum 2.pdf").exists()

    r2 = client.post("/upload", data={"bid_id": "Bid3"}, files=files[:1])
    assert r2.json()["saved"][0]["already_existed"] is True
    assert r2.json()["report"]["files"][0]["status"] == "skipped_unchanged"

    assert client.post("/upload", data={"bid_id": "../x"}, files=files[:1]).status_code == 400
