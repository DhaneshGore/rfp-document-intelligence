"""Streamlit UI for the RFP search engine. Talks to the FastAPI service over HTTP.

Run:  python main.py up        (API + UI)
  or: python main.py serve  &  streamlit run app.py
"""
from __future__ import annotations

import sys
import time

import httpx
import streamlit as st

from config import get_settings

S = get_settings()
API = S.api_url.rstrip("/")
DOC_TYPES = ["", "rfp", "addendum", "bid_page", "specs", "affidavit", "other"]
MODES = ["hybrid_rerank", "hybrid", "vector", "bm25"]

st.set_page_config(page_title="RFP Intelligence - Search", layout="wide")


def _ui_error(msg: str) -> RuntimeError:
    """Errors are shown in the page and also printed to the terminal running the UI."""
    print(f"UI ERROR: {msg}", file=sys.stderr, flush=True)
    return RuntimeError(msg)


def api(method: str, path: str, timeout: float = 60, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", timeout=timeout, **kw)
    except httpx.HTTPError as exc:
        raise _ui_error(f"Cannot reach the API at {API} ({exc}). Start it with `python main.py serve`.")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise _ui_error(f"API error {r.status_code} on {method} {path}: {detail}")
    return r.json()


@st.cache_data(ttl=10, show_spinner=False)
def load_bids() -> list[dict]:
    return api("GET", "/bids")["bids"]


@st.cache_data(ttl=300, show_spinner=False)
def load_models() -> dict:
    return api("GET", "/models", timeout=30)


def start_job(body: dict, path: str = "/index/jobs") -> None:
    """Submit a background job (indexing, or extraction via /extract/jobs); its progress
    panel is rendered in the sidebar."""
    job = api("POST", path, json=body)
    st.session_state["active_job"] = job["job_id"]
    st.session_state["active_job_kind"] = job.get("kind", "index")
    st.session_state.pop("last_job", None)


def fmt_duration(seconds: float) -> str:
    return f"~{seconds / 60:.0f} min" if seconds >= 90 else f"~{seconds:.0f} s"


def render_job_panel(poll_s: float = 1.0) -> None:
    """Sidebar panel for the active job: status, progress, Cancel button. Runs last in the
    script so the rest of the page is already drawn while it polls."""
    job_id = st.session_state.get("active_job")
    if not job_id:
        return
    extract = st.session_state.get("active_job_kind") == "extract"
    box = st.sidebar.container(border=True)
    box.markdown("**Extraction job**" if extract else "**Indexing job**")
    if box.button("Cancel extraction" if extract else "Cancel indexing", key=f"cancel_{job_id}", type="secondary"):
        try:
            api("POST", f"/index/jobs/{job_id}/cancel", timeout=10)
        except RuntimeError as exc:
            box.error(str(exc))
    bar = box.progress(0.0, text="...")
    note = box.empty()
    while True:
        try:
            job = api("GET", f"/index/jobs/{job_id}", timeout=10)
        except RuntimeError as exc:  # e.g. API restarted: the in-memory job is gone
            note.error(str(exc))
            st.session_state.pop("active_job", None)
            st.session_state.pop("active_job_kind", None)
            return
        prog = job["progress"]
        total, done = prog.get("total") or 0, prog.get("done") or 0
        status = job["status"]
        if status == "queued":
            ahead = job.get("queue_position", 0)
            bar.progress(0.0, text=f"Queued - waiting for {ahead} earlier job(s) to finish")
        elif status in ("running", "cancelling"):
            label = f"{prog.get('current') or 'starting'} ({min(done + 1, total or 1)}/{total or '?'})"
            bar.progress(min(done / total, 1.0) if total else 0.0, text=label)
            if extract:
                detail = prog.get("detail") or "waiting for the first model response (can take a minute)"
                cancelling = "Cancelling before the next model call... "
            else:
                detail = prog.get("detail") or "large PDFs can take minutes"
                cancelling = "Cancelling at the next page batch... "
            note.caption((cancelling if status == "cancelling" else "") + detail)
        else:  # finished
            st.session_state["last_job"] = job
            st.session_state.pop("active_job", None)
            st.session_state.pop("active_job_kind", None)
            load_bids.clear()
            st.rerun()
        time.sleep(poll_s)


def show_job_result(job: dict) -> None:
    status = job["status"]
    if status == "cancelled":
        st.warning(f"Indexing cancelled. {job.get('error') or ''}")
    elif status == "failed":
        st.error(f"Indexing job failed: {job.get('error')}")
    rep = job.get("report") or {}
    for f in rep.get("files", []):
        msg = f"{f['file_name']}: {f['status']} - {f['message']}"
        if f["status"] == "indexed":
            st.success(msg)
        elif f["status"].startswith("skipped") or f["status"] == "removed":
            st.info(msg)
        else:
            st.error(msg)
        for w in f.get("warnings", []):
            st.warning(f"{f['file_name']}: {w}")
    if rep:
        st.caption(f"Job finished in {rep.get('seconds')} s · chunks added: {rep.get('total_chunks_added')}")


def show_extract_result(job: dict) -> None:
    status = job["status"]
    if status == "cancelled":
        st.warning(f"Extraction cancelled. {job.get('error') or ''}")
        return
    if status == "failed":
        st.error(f"Extraction failed: {job.get('error')}")
        return
    res = job.get("report") or {}
    v = res.get("validation", {})
    st.success(f"Extraction of **{res.get('bid_id')}** finished: {v.get('passed')} verified, {v.get('failed')} "
               f"failed (unverified {v.get('unverified')}, error {v.get('error')}), {v.get('not_found')} not found · "
               f"{res.get('llm_calls')} LLM calls · model {res.get('model')} · run {res.get('run_id')}")
    st.markdown("**Output files**\n\n" + "\n".join(f"- {k}: `{p}`" for k, p in (res.get("outputs") or {}).items()))
    if res.get("backup_dir"):
        st.caption(f"Previous outputs backed up (verified identical) in `{res['backup_dir']}`")
    for e in res.get("errors") or []:
        st.warning(e)
    st.dataframe([{k: f.get(k) for k in ("field", "status", "confidence", "value", "sources")}
                  for f in res.get("fields", [])], use_container_width=True, hide_index=True)
    changes = res.get("addendum_changes") or []
    st.markdown("**Addendum changes**" + ("" if changes else ": none recorded"))
    for c in changes:
        st.write(f"- {c['field']}: `{c.get('previous_value')}` → `{c.get('value')}` (Addendum {c.get('addendum_number')})")


def page_label(r: dict) -> str:
    p, e = r.get("page_number"), r.get("page_end")
    if p is None:
        return "n/a (web page)"
    return f"{p}-{e}" if e and e != p else str(p)


def show_result(r: dict, idx: int | None = None, nested: bool = False) -> None:
    head = f"**#{r['rank']}** · `{r['file_name']}` · page {page_label(r)} · {r['bid_id']} · {r['doc_type']}"
    st.markdown(head)
    caption = (
        f"chunk_id: {r['chunk_id']} | section: {r.get('section') or '-'} | "
        f"scores: {r.get('scores')} | ranks: {r.get('ranks')}"
    )
    if nested:  # Streamlit forbids expanders inside expanders
        st.text(r["text"])
        st.caption(caption)
        return
    with st.expander(f"Passage ({r['citation']})", expanded=idx is not None and idx < 2):
        st.text(r["text"])
        st.caption(caption)


# ----------------------------------------------------------------------- sidebar
st.sidebar.title("RFP Search")
try:
    health = api("GET", "/health", timeout=10)
    st.sidebar.success(f"API online · {health['bids']} bids · {health['documents']} docs · {health['chunks']} chunks")
    bids = load_bids()
except RuntimeError as exc:
    st.sidebar.error(str(exc))
    st.error(str(exc))
    st.stop()

bid_ids = [b["bid_id"] for b in bids]
ALL_BIDS, ALL_TYPES = "All bids", "All types"
f_bid = st.sidebar.selectbox("Bid filter", [ALL_BIDS] + bid_ids)
f_bid = "" if f_bid == ALL_BIDS else f_bid
f_type = st.sidebar.selectbox("Document type filter", [ALL_TYPES] + DOC_TYPES[1:])
f_type = "" if f_type == ALL_TYPES else f_type
f_add = st.sidebar.number_input("Addendum number (0 = any)", min_value=0, max_value=50, value=0, step=1)
mode = st.sidebar.selectbox("Retrieval mode", MODES)
top_k = st.sidebar.slider("Results (top_k)", 1, 20, 5)

try:
    models_info = load_models()
except RuntimeError as exc:
    models_info = {"models": [], "default": None, "api_key_configured": False}
    st.sidebar.warning(str(exc))
usable = [m for m in models_info["models"] if m["usable"]]
model_ids = [m["id"] for m in usable]
default_idx = model_ids.index(models_info["default"]) if models_info.get("default") in model_ids else 0
model = st.sidebar.selectbox(
    "LLM (OpenRouter free models)", model_ids, index=default_idx if model_ids else None
)
if model:
    st.sidebar.caption(f"Free status: {next(x['status'] for x in usable if x['id'] == model)}")
if not models_info.get("api_key_configured"):
    st.sidebar.warning("OPENROUTER_API_KEY not set: search works, Q&A answers are disabled.")

params = {k: v for k, v in {"bid_id": f_bid, "doc_type": f_type, "addendum_number": f_add or None}.items() if v}

tab_search, tab_ask, tab_upload, tab_docs, tab_extract = st.tabs(
    ["🔎 Search", "💬 Ask", "⬆️ Upload & index", "📚 Indexed documents", "🧾 Extract Bid Fields"])

# ----------------------------------------------------------------------- search
with tab_search:
    q = st.text_input("Search query", placeholder="e.g. submission deadline, bid bond, WD22TB4, warranty")
    if st.button("Search", type="primary") and q.strip():
        with st.spinner("Searching..."):
            try:
                res = api("GET", "/search", params={"q": q, "top_k": top_k, "mode": mode, **params})
            except RuntimeError as exc:
                st.error(str(exc))
                res = None
        if res is not None:
            if not res["results"]:
                st.info("No matching passages.")
            for i, r in enumerate(res["results"]):
                show_result(r, i)

# ----------------------------------------------------------------------- ask
with tab_ask:
    question = st.text_area("Question", placeholder="What is the submission deadline for Bid1 after all addendums?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner(f"Retrieving evidence and asking {model}..."):
            try:
                ans = api("POST", "/ask", timeout=S.llm_timeout_s * 4,
                          json={"question": question, "model": model, "mode": mode, **params})
            except RuntimeError as exc:
                st.error(str(exc))
                ans = None
        if ans is not None:
            status = ans.get("status") or ("verified" if ans.get("found") and not ans.get("error") else "unverified")
            if status == "verified":
                st.markdown("**✅ Verified answer** – every value is cited and found in the cited passages.")
                st.success(ans["answer"])
            elif status == "unverified":
                st.markdown("**⚠️ Unverified answer** – check it against the evidence before relying on it.")
                st.warning(ans.get("error") or "Citation or grounding check failed.")
                st.info(ans["answer"])
            elif status == "incomplete":
                st.error(ans.get("error") or "Incomplete response.")
                if ans.get("answer"):
                    with st.expander("Partial model output (not an answer)"):
                        st.text(ans["answer"])
            elif status == "not_found":
                st.info(ans.get("answer") or "The information was not found in the indexed documents.")
            else:  # error
                st.error(ans.get("error") or "The request failed.")
            if ans.get("citations") and status in ("verified", "unverified"):
                st.subheader("Citations")
                for c in ans["citations"]:
                    pg = f"p. {c['page_number']}" if c.get("page_number") else "web page"
                    with st.expander(f"[{c['ref']}] {c['file_name']} · {pg} · {c['bid_id']} · {c['doc_type']}"):
                        st.text(c["text"])
                        st.caption(f"chunk_id: {c['chunk_id']}")
            if ans.get("invalid_citation_refs"):
                st.warning(f"Model cited non-existent passages: {ans['invalid_citation_refs']}")
            if ans.get("model"):
                st.caption(f"model: {ans['model']} · status: {status} · finish: {ans.get('finish_reason')} · "
                           f"latency: {ans.get('latency_s')}s · usage: {ans.get('usage')}")
            if ans.get("evidence"):
                with st.expander(f"All retrieved evidence ({len(ans['evidence'])} passages)"):
                    for r in ans["evidence"]:
                        show_result(r, nested=True)

# ----------------------------------------------------------------------- upload
with tab_upload:
    st.write("Upload PDF or HTML files into an existing bid or a new one. Original bid folders are never modified; "
             f"uploads are stored under `data/uploads/<bid_id>/`. Max {S.max_upload_mb} MB per file.")
    choice = st.radio("Target bid", ["Existing bid", "New bid"], horizontal=True)
    if choice == "Existing bid" and bid_ids:
        target = st.selectbox("Bid", bid_ids)
    else:
        target = st.text_input("New bid id (letters, digits, - or _)", placeholder="Bid3")
    files = st.file_uploader("Documents", type=["pdf", "html", "htm"], accept_multiple_files=True)
    do_index = st.checkbox("Index after upload", value=True)
    busy = bool(st.session_state.get("active_job"))
    if busy:
        st.info("An indexing job is running - see the sidebar. New uploads will queue behind it.")
    if st.button("Upload", type="primary", disabled=not files or not target):
        try:
            payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
            up = api("POST", "/upload", timeout=120, data={"bid_id": target, "index": "false"}, files=payload)
        except RuntimeError as exc:
            st.error(str(exc))
            up = None
        if up:
            for r in up["rejected"]:
                st.error(f"Rejected {r['original_name']}: {r['error']}")
            for s_ in up["saved"]:
                info = s_.get("pdf") or {}
                extra = (f" · {info['pages']} pages, {info['images']} images" if "pages" in info else "")
                st.write(f"Saved `{s_['original_name']}` as `{s_['stored_as']}`{extra}"
                         + (" (identical file already uploaded)" if s_["already_existed"] else ""))
            if do_index and up["saved"]:
                large = [s_ for s_ in up["saved"] if (s_.get("pdf") or {}).get("large")]
                if large:
                    # Ask before indexing: decision UI below survives the rerun via session state
                    st.session_state["pending_index"] = {"bid_id": up["bid_id"], "saved": up["saved"],
                                                         "large": large}
                else:
                    try:
                        start_job({"bid_id": up["bid_id"], "paths": [s_["path"] for s_ in up["saved"]]})
                    except RuntimeError as exc:
                        st.error(str(exc))

    pending = st.session_state.get("pending_index")
    if pending:
        lines = []
        for s_ in pending["large"]:
            p = s_["pdf"]
            lines.append(f"- **{s_['stored_as']}**: {p['pages']} pages, {p['images']} images, "
                         f"{p['pages_without_text']} pages without a text layer - estimated "
                         f"**{fmt_duration(p['estimated_seconds_with_ocr'])} with OCR** on this machine")
        st.warning(
            "Large PDF detected - indexing runs one job at a time, so this will delay other uploads.\n\n"
            + "\n".join(lines)
            + "\n\nWithout OCR indexing is faster, but text inside images and scanned pages will not be searchable."
        )
        paths = [s_["path"] for s_ in pending["saved"]]
        c1, c2, c3 = st.columns(3)
        try:
            if c1.button("Index with OCR", type="primary"):
                st.session_state.pop("pending_index")
                start_job({"bid_id": pending["bid_id"], "paths": paths, "ocr": True})
                st.rerun()
            if c2.button("Index without OCR (faster)"):
                st.session_state.pop("pending_index")
                start_job({"bid_id": pending["bid_id"], "paths": paths, "ocr": False})
                st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
        if c3.button("Don't index now"):
            st.session_state.pop("pending_index")
            st.info("Files are saved but not indexed. Use 'Re-index all' later to index them.")
            st.rerun()

    if st.session_state.get("last_job") and st.session_state["last_job"].get("kind", "index") == "index":
        show_job_result(st.session_state["last_job"])

# ----------------------------------------------------------------------- docs
with tab_docs:
    c1, c2 = st.columns([1, 3])
    if c1.button("Re-index all (incremental)"):
        try:
            start_job({"all": True})
            c2.info("Indexing started - progress and Cancel are in the sidebar.")
        except RuntimeError as exc:
            st.error(str(exc))
    try:
        docs = api("GET", "/documents", params={"bid_id": f_bid} if f_bid else None)["documents"]
        rows = [{k: d.get(k) for k in ("bid_id", "file_name", "doc_type", "addendum_number", "document_date",
                                        "num_pages", "chunks", "status", "message")} for d in docs]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    except RuntimeError as exc:
        st.error(str(exc))

# ----------------------------------------------------------------------- extract
with tab_extract:
    st.write("Generate the structured JSON for one indexed bid: the 20 Section 8 fields with citations, confidence, "
             "notes, validation and addendum changes. Nothing runs until you confirm and click the button.")
    x_bids = [b["bid_id"] for b in bids if b.get("documents")]
    pre = None
    if not x_bids:
        st.info("No indexed bids yet - upload and index documents first.")
    else:
        x_bid = st.selectbox("Bid to extract", x_bids, key="x_bid")
        try:
            pre = api("GET", "/extract/preflight", params={"bid_id": x_bid}, timeout=30)
        except RuntimeError as exc:
            st.error(str(exc))
    if pre:
        st.markdown(f"**Documents indexed under `{pre['bid_id']}`** - only these are searched; passages from other "
                    "bids are never used.")
        st.dataframe([{k: d.get(k) for k in ("file_name", "doc_type", "addendum_number", "document_date", "chunks",
                                              "status")} for d in pre["documents"]],
                     use_container_width=True, hide_index=True)
        if pre["addenda"]:
            st.caption("Addenda detected: " + "; ".join(
                f"{a['file_name']} (Addendum {a['addendum_number'] or '?'})" for a in pre["addenda"])
                + ". Reconciliation compares their values with the original documents and keeps the newer one.")
        else:
            st.info("No addendum detected for this bid. If you uploaded one, check its type under 'Indexed "
                    "documents': the file name or first page must say 'Addendum' or 'Amendment'.")
        for w in pre["warnings"]:
            st.warning(w)

        st.markdown("**Output files**\n\n" + "\n".join(
            f"- `{o['path']}`" + (f" - exists, modified {o['modified_utc']}" if o["exists"] else "")
            for o in pre["outputs"].values()))
        x_models = [m for m in model_ids if m in pre["models"]] or pre["models"]
        x_default = x_models.index(pre["default_model"]) if pre["default_model"] in x_models else 0
        x_model = st.selectbox("Extraction model (OpenRouter free models)", x_models, index=x_default,
                               key="x_model")
        est = pre["estimate"]
        st.warning(f"Extraction sends about **{est['min_requests']}-{est['max_requests']} requests** to OpenRouter "
                   "(free models allow 50 requests per day per account) and usually takes 5-20 minutes. The run "
                   "stops at the first rate-limit, quota, authentication or payment error and does not retry it.")
        if not pre["api_key_configured"]:
            st.error("OPENROUTER_API_KEY is not set - extraction needs the LLM.")
        overwrite_ok = True
        if pre["existing"]:
            st.warning("Output files for this bid already exist. They will be copied to "
                       f"`{pre['backup_dir_pattern']}` and checked byte for byte before the run; if the backup fails "
                       "the run does not start.")
            overwrite_ok = st.checkbox("Back up and overwrite the existing output files", key=f"x_over_{x_bid}")
        confirm = st.checkbox(f"Run extraction for **{pre['bid_id']}** → `{pre['outputs']['json']['path']}` "
                              f"with `{x_model}`", key=f"x_confirm_{x_bid}")
        if st.session_state.get("active_job"):
            st.info("A job is running - see the sidebar. A new extraction will queue behind it.")
        ready = confirm and overwrite_ok and bool(x_model) and pre["api_key_configured"]
        if st.button("Extract Bid Fields", type="primary", disabled=not ready):
            try:
                start_job({"bid_id": pre["bid_id"], "model": x_model, "confirm_overwrite": bool(pre["existing"])},
                          path="/extract/jobs")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
    if st.session_state.get("last_job") and st.session_state["last_job"].get("kind") == "extract":
        show_extract_result(st.session_state["last_job"])

# ----------------------------------------------------------------------- job panel
# Rendered last: it polls until the job finishes, after the rest of the page is drawn.
render_job_panel()
