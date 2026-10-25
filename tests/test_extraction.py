"""Structured extraction (Section 8) - offline: fake embedder, stub LLMs, no network or quota."""
import json
import re

import pytest

from extraction.fields import FIELD_NAMES, FIELDS, GROUPS, fields_in
from extraction.schemas import BidExtraction
from extraction.service import ExtractionService, parse_json_object, score_confidence, write_extraction
from schemas import LLMResponse, LLMUsage, SearchResult
from services.llm_service import LLMTimeoutError
from tests.conftest import make_pdf

ASSIGNMENT_FIELDS = [
    "Bid Number", "Title", "Due Date", "Bid Submission Type", "Term of Bid", "Pre Bid Meeting", "Installation",
    "Bid Bond Requirement", "Delivery Date", "Payment Terms", "Any Additional Documentation Required",
    "MFG for Registration", "Contract or Cooperative to use", "Model_no", "Part_no", "Product", "contact_info",
    "company_name", "Bid Summary", "Product Specification",
]


def _blocks(user: str) -> dict[int, str]:
    body = user.split("Evidence passages:\n\n", 1)[1].rsplit("\n\nReturn the JSON", 1)[0]
    out = {}
    for b in re.split(r"\n\n(?=\[\d+\] \()", body):
        n = int(b[1:b.index("]")])
        out[n] = b
    return out


class NeedleLLM:
    """Deterministic extractor: each field gets a fixed value and cites the passages that
    contain its needles. Fields without an answer (or whose needles are in no passage) are null.
    `overrides` replaces a field's whole entry (to simulate bad model behaviour)."""

    def __init__(self, answers=None, overrides=None, text=None, exc=None, finish_reason="stop", scoped=None,
                 change_types=None):
        self.answers, self.overrides = answers or {}, overrides or {}
        self.text, self.exc, self.finish_reason = text, exc, finish_reason
        self.scoped = scoped or {}  # {"original"|"addendum": {field: (value, needles)}} for reconciliation calls
        # addendum scope: the "change" label per field (default "changed" for values, "none" for nulls)
        self.change_types = change_types or {}
        self.calls: list[dict] = []

    @staticmethod
    def scope_of(user: str):
        if "come from the ORIGINAL" in user:
            return "original"
        if "come from ADDENDUMS only" in user:
            return "addendum"
        return None

    def complete(self, messages, model=None):
        user = messages[1]["content"]
        keys = json.loads(user.rsplit("exactly these keys: ", 1)[1])
        blocks = _blocks(user)
        scope = self.scope_of(user)
        self.calls.append({"keys": keys, "passages": blocks, "scope": scope})
        if self.exc:
            raise self.exc
        if self.text is not None:
            return self._resp(self.text)
        answers = {**self.answers, **self.scoped.get(scope, {})}
        out = {}
        for k in keys:
            if k in self.overrides:
                out[k] = self.overrides[k]
                continue
            value, needles = answers.get(k, (None, []))
            refs = []
            for nd in needles:
                n = next((n for n, t in blocks.items() if nd in t), None)
                if n and n not in refs:
                    refs.append(n)
            if value is None or not refs:
                out[k] = {"value": None, "refs": [], "notes": "Not found in documents"}
            else:
                out[k] = {"value": value, "refs": refs, "notes": "", "confidence": 0.99}  # model confidence ignored
            if scope == "addendum":
                out[k]["change"] = self.change_types.get(k, "changed" if out[k]["value"] is not None else "none")
        return self._resp(json.dumps(out))

    def _resp(self, text):
        return LLMResponse(text=text, model="stub/extractor:free", latency_s=0.0, usage=LLMUsage(total_tokens=10),
                           finish_reason=self.finish_reason)


# Values for the shared `bid_folder` fixture (BidA): RFP, addendum, specs, affidavit, HTML page.
ANSWERS = {
    "Bid Number": ("RFQ-555", ["Solicitation Number: RFQ-555"]),
    "Title": ("RFQ-555 - Widget Supply", ["RFQ-555 - Widget Supply"]),
    "company_name": ("City of Springfield", ["City of Springfield"]),
    "Due Date": ("April 15, 2024 at 2:00 PM", ["extended to April 15, 2024"]),
    "Bid Submission Type": ("Submit electronically through the portal", ["Submit electronically"]),
    "Bid Bond Requirement": ("A bid bond of 5% of the total bid amount is required", ["bid bond of 5%"]),
    "Model_no": (["WX-100"], ["Model WX-100"]),
    "Part_no": (["210-ABCD"], ["210-ABCD"]),
    "Product": (["Blue widget WX-100 | 40"], ["Blue widget WX-100 | 40"]),
    "Product Specification": ("16 GB memory, 512 GB SSD; three year manufacturer warranty",
                              ["16 GB memory", "three year manufacturer warranty"]),
    "Any Additional Documentation Required": (["Affidavit that the business is registered in the State"],
                                              ["registered in the State"]),
    "Bid Summary": ("The City of Springfield requests proposals for widget supply under RFQ-555. "
                    "Proposals are due April 15, 2024 at 2:00 PM and a bid bond of 5% is required.",
                    ["City of Springfield", "extended to April 15, 2024", "bid bond of 5%"]),
}
# What the original RFP said before Addendum 1 (used by the reconciliation "original" call).
ORIGINAL = {"Due Date": ("April 1, 2024 at 2:00 PM", ["Proposals are due April 1, 2024"])}


@pytest.fixture
def indexed(engine, bid_folder):
    engine.indexing.index_folder(bid_folder)
    return engine


def service(engine, llm, **kw):
    kw.setdefault("per_query", 10)
    kw.setdefault("mode", "hybrid")
    return ExtractionService(engine.retrieval, llm, **kw)


# ----------------------------------------------------------------------- registry and schema
def test_registry_has_the_20_assignment_fields_in_order():
    assert list(FIELD_NAMES) == ASSIGNMENT_FIELDS
    assert all(f.queries and f.description for f in FIELDS)
    assert {f.group for f in FIELDS} == set(GROUPS) and all(fields_in(g) for g in GROUPS)


def test_full_extraction_record(indexed):
    llm = NeedleLLM(ANSWERS, scoped={"original": ORIGINAL})
    rec = service(indexed, llm).extract_bid("BidA")
    # 5 group calls + reconciliation: 3 addendum-scope calls (one per field group with addendum-sensitive
    # fields) + 1 original-scope call (only Due Date has a substantive addendum change); no retries
    assert list(rec.fields) == ASSIGNMENT_FIELDS and rec.llm_calls == len(llm.calls) == len(GROUPS) + 4
    assert [c["scope"] for c in llm.calls[-4:]] == ["addendum", "addendum", "addendum", "original"]
    assert llm.calls[-1]["keys"] == ["Due Date"]  # originals requested only for the changed field
    for name in ANSWERS:
        f = rec.fields[name]
        assert f.status == "verified", (name, f.notes)
        assert f.sources and all(s.file and s.chunk_id and s.citation for s in f.sources)
        assert 0.5 < f.confidence <= 1.0
    due = rec.fields["Due Date"]
    assert due.value == "April 15, 2024 at 2:00 PM" and due.sources[0].file == "Addendum 1 RFP-555.pdf"
    assert due.sources[0].page == 1 and due.sources[0].doc_type == "addendum"
    (change,) = rec.addendum_changes
    assert change.field == "Due Date" and change.addendum_number == 1 and change.value == due.value
    assert change.previous_value == "April 1, 2024 at 2:00 PM"
    assert change.previous_sources[0].file == "RFP-555 Widget Supply.pdf" and change.previous_sources[0].page == 1
    assert "Changed by Addendum 1" in due.notes
    html = rec.fields["company_name"].sources[0]
    assert html.file.endswith(".html") and html.page is None  # HTML has no page numbers
    for name in set(ASSIGNMENT_FIELDS) - set(ANSWERS):
        f = rec.fields[name]
        assert f.status == "not_found" and f.value is None and f.sources == []
        assert f.notes.startswith("Not found in documents")
    v = rec.validation
    assert (v.passed, v.not_found, v.failed) == (len(ANSWERS), 20 - len(ANSWERS), 0)


def test_output_is_valid_json_in_section_8_1_shape(indexed, tmp_path):
    rec = service(indexed, NeedleLLM(ANSWERS)).extract_bid("BidA")
    path = write_extraction(rec, tmp_path / "out")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "BidA.json" and data["bid_id"] == "BidA"
    assert set(data) >= {"bid_id", "fields", "addendum_changes", "validation"}
    assert set(data["validation"]) >= {"passed", "failed", "not_found"}
    assert list(data["fields"]) == ASSIGNMENT_FIELDS
    for f in data["fields"].values():
        assert set(f) >= {"value", "sources", "confidence", "notes", "status"}
        assert f["status"] in {"verified", "unverified", "not_found", "error"} and 0 <= f["confidence"] <= 1
        assert all(set(s) >= {"file", "page", "chunk_id"} for s in f["sources"])
    assert data["fields"]["Payment Terms"]["value"] is None
    BidExtraction.model_validate(data)  # round-trips


# ----------------------------------------------------------------------- evidence validation
def test_fabricated_or_unknown_refs_are_rejected(indexed):
    llm = NeedleLLM(ANSWERS, overrides={
        "Term of Bid": {"value": "Five years with two renewals", "refs": [99], "notes": ""},
        "Payment Terms": {"value": "Net 30", "refs": ["passage 2"], "notes": ""},
        "Bid Bond Requirement": {"value": "Not required", "refs": [], "notes": "", "confidence": 1.0},
    })
    rec = service(indexed, llm).extract_bid("BidA")
    for name in ("Term of Bid", "Payment Terms", "Bid Bond Requirement"):
        f = rec.fields[name]
        assert f.status == "unverified" and f.value is None and f.sources == [] and f.confidence == 0.0
        assert "cites no valid evidence passage" in f.notes


def test_mixed_valid_and_fabricated_refs_stay_unverified(indexed):
    class Mixed(NeedleLLM):
        def complete(self, messages, model=None):
            resp = super().complete(messages, model)
            obj = json.loads(resp.text)
            if "Model_no" in obj and obj["Model_no"]["refs"]:
                obj["Model_no"]["refs"].append(42)
            return self._resp(json.dumps(obj))

    f = service(indexed, Mixed(ANSWERS)).extract_bid("BidA").fields["Model_no"]
    assert f.status == "unverified" and f.value == ["WX-100"] and "fabricated" in f.notes and f.confidence <= 0.5


@pytest.mark.parametrize("field,entry_fn,problem", [
    # value cited to a passage that does not contain it
    ("Bid Number", lambda b: {"value": "RFQ-555", "refs": [n for n, t in b.items() if "AFFIDAVIT" in t][:1]},
     "not found verbatim"),
    # a number that the cited passage does not contain
    ("Due Date", lambda b: {"value": "April 16, 2024 at 2:00 PM",
                            "refs": [n for n, t in b.items() if "extended to April 15" in t][:1]}, "numbers"),
    # an email that is not in the passage
    ("contact_info", lambda b: {"value": "Jane Doe, jane@springfield.gov",
                                "refs": [n for n, t in b.items() if "City of Springfield" in t][:1]}, "contact"),
    # mostly invented wording
    ("Payment Terms", lambda b: {"value": "Net 45 after receipt of a correct invoice",
                                 "refs": [n for n, t in b.items() if "bid bond" in t][:1]}, "numbers"),
])
def test_values_not_supported_by_cited_evidence_are_unverified(indexed, field, entry_fn, problem):
    class Bad(NeedleLLM):
        def complete(self, messages, model=None):
            resp = super().complete(messages, model)
            obj = json.loads(resp.text)
            if field in obj:
                obj[field] = {**entry_fn(self.calls[-1]["passages"]), "notes": ""}
            return self._resp(json.dumps(obj))

    f = service(indexed, Bad(ANSWERS)).extract_bid("BidA").fields[field]
    assert f.status == "unverified" and f.sources and f.confidence <= 0.5
    assert problem in f.notes, f.notes


def test_confidence_formula_is_evidence_based():
    assert score_confidence(1.0, 1.0, True, True, 1.0, verified=True) == 1.0
    assert score_confidence(1.0, 1.0, False, True, 1.0, verified=True) == 0.65  # ungrounded numbers drop support
    assert score_confidence(0.5, 1.0, True, True, 1.0, verified=False) == 0.5  # unverified is capped
    assert score_confidence(1.0, 0.5, True, False, 0.25, verified=True) == 0.56


# ----------------------------------------------------------------------- failure handling
@pytest.mark.parametrize("llm,needle", [
    (NeedleLLM(text="Sure! Here are the fields: Bid Number is RFQ-555."), "invalid JSON"),
    (NeedleLLM(text='{"Bid Number": {"value": "RFQ-555", "refs": [1]'), "invalid JSON"),
    (NeedleLLM(text="[1, 2, 3]"), "invalid JSON"),
    (NeedleLLM(exc=LLMTimeoutError("timeout after 60s")), "LLMTimeoutError"),
    (NeedleLLM(ANSWERS, finish_reason="length"), "token limit"),
])
def test_bad_model_output_is_an_error_not_a_crash(indexed, llm, needle):
    rec = service(indexed, llm).extract_bid("BidA")
    assert list(rec.fields) == ASSIGNMENT_FIELDS
    assert all(f.status == "error" and f.value is None and f.confidence == 0.0 and needle in f.notes
               for f in rec.fields.values())
    assert rec.validation.error == 20 and rec.validation.passed == 0


def test_fenced_json_and_omitted_or_malformed_fields(indexed):
    assert parse_json_object('<think>x</think>```json\n{"a": 1}\n```') == {"a": 1}
    llm = NeedleLLM(ANSWERS, overrides={"Title": "RFQ-555 - Widget Supply"})

    class Omit(NeedleLLM):
        def complete(self, messages, model=None):
            resp = super().complete(messages, model)
            obj = json.loads(resp.text)
            obj.pop("company_name", None)
            return self._resp(json.dumps(obj))

    rec = service(indexed, llm).extract_bid("BidA")
    assert rec.fields["Title"].status == "error" and "Malformed" in rec.fields["Title"].notes
    rec = service(indexed, Omit(ANSWERS)).extract_bid("BidA")
    assert rec.fields["company_name"].status == "error" and "omitted" in rec.fields["company_name"].notes


def test_empty_retrieval_never_invents_values(engine):
    llm = NeedleLLM(ANSWERS)
    rec = service(engine, llm).extract_bid("NoDocsBid")
    assert llm.calls == [] and rec.llm_calls == 0
    assert all(f.status == "not_found" and f.value is None and "no passages retrieved" in f.notes
               for f in rec.fields.values())


# ----------------------------------------------------------------------- bid isolation / unseen bids
def test_one_bids_evidence_never_reaches_another(indexed, settings):
    other = settings.bids_root / "BidB"
    other.mkdir()
    make_pdf(other / "RFP-777 Laptops.pdf", ["Request for Proposal RFP-777\nIssued by the County of Shelbyville.\n"
                                            "Proposals are due June 3, 2024 at 11:00 AM."])
    indexed.indexing.index_folder(other)
    llm = NeedleLLM({**ANSWERS, "Bid Number": ("RFP-777", ["RFP-777"])})
    rec = service(indexed, llm).extract_bid("BidB")
    passages = [t for c in llm.calls for t in c["passages"].values()]
    assert passages and all("bid: BidB;" in t for t in passages)
    assert not any("Springfield" in t or "WX-100" in t for t in passages)
    assert rec.fields["Bid Number"].value == "RFP-777" and rec.fields["Bid Number"].status == "verified"
    assert rec.fields["company_name"].value is None  # BidA's answer cannot be cited in BidB
    assert all(s.file == "RFP-777 Laptops.pdf" for f in rec.fields.values() for s in f.sources)


def test_foreign_passages_from_a_faulty_retriever_are_dropped():
    def r(i, bid):
        return SearchResult(rank=i, score=1.0, chunk_id=f"{bid}-{i}", text=f"Solicitation Number: X-{i}00 for {bid}",
                            bid_id=bid, file_name=f"{bid}.pdf", doc_type="rfp", page_number=1, page_end=1,
                            source_path=f"/{bid}.pdf", citation=f"{bid}.pdf, p. 1")

    class LeakyRetrieval:
        def search(self, query, top_k=5, filters=None, mode=None):
            return [r(1, "OtherBid"), r(2, filters.bid_id)]

    llm = NeedleLLM({"Bid Number": ("X-200", ["X-200"])})
    rec = ExtractionService(LeakyRetrieval(), llm).extract_bid("Mine")
    assert all("OtherBid" not in t for c in llm.calls for t in c["passages"].values())
    assert rec.fields["Bid Number"].status == "verified" and rec.fields["Bid Number"].sources[0].file == "Mine.pdf"
    assert "other bids discarded" in rec.fields["Title"].notes


def test_unfamiliar_bid_folder_with_new_names(engine, settings):
    d = settings.bids_root / "Harbor-Authority_2025-Q3"
    d.mkdir()
    make_pdf(d / "Solicitation Package.pdf", [
        "INVITATION FOR BID IFB 25-0412\nPort of Harbor Authority\nProcurement contact: Ana Ruiz, "
        "ana.ruiz@harbor.example, 555-201-3344\nSealed bids are due September 9, 2025 at 3:00 PM PT.",
        "SCOPE\nSupply and install 12 rugged tablets, model RT-9X, part RT9X-128.",
    ])
    make_pdf(d / "Amendment 2 to IFB 25-0412.pdf", ["AMENDMENT No. 2\nThe bid due date is changed to "
                                                   "September 16, 2025 at 3:00 PM PT."])
    engine.indexing.index_folder(d)
    bid = d.name
    llm = NeedleLLM(scoped={"original": {"Due Date": ("September 9, 2025 at 3:00 PM PT", ["due September 9, 2025"])}},
                    answers={
        "Bid Number": ("IFB 25-0412", ["INVITATION FOR BID IFB 25-0412"]),
        "company_name": ("Port of Harbor Authority", ["Port of Harbor Authority"]),
        "contact_info": ("Ana Ruiz, ana.ruiz@harbor.example, 555-201-3344", ["ana.ruiz@harbor.example"]),
        "Due Date": ("September 16, 2025 at 3:00 PM PT", ["changed to September 16, 2025"]),
        "Model_no": (["RT-9X"], ["model RT-9X"]),
        "Part_no": (["RT9X-128"], ["RT9X-128"]),
        "Installation": ("Supply and install 12 rugged tablets", ["Supply and install"]),
    })
    rec = service(engine, llm).extract_bid(bid)
    for name in ("Bid Number", "company_name", "contact_info", "Due Date", "Model_no", "Part_no", "Installation"):
        assert rec.fields[name].status == "verified", (name, rec.fields[name].notes)
    assert rec.fields["Due Date"].sources[0].file == "Amendment 2 to IFB 25-0412.pdf"
    (change,) = rec.addendum_changes
    assert change.addendum_number == 2 and change.previous_value == "September 9, 2025 at 3:00 PM PT"
    assert change.previous_sources[0].file == "Solicitation Package.pdf"
    assert rec.bid_id == bid and rec.validation.passed == 7 and rec.validation.not_found == 13
