"""Evidence selection and reconciliation robustness - offline (fake embedder, stub LLMs).

Fixtures use invented bids (no Bid1/Bid2 values), so the checks are about generic
behaviour: a named buyer, a form required "with the response" and every product tier must
reach the model; large reconciliation outputs and partial failures must not lose changes.
"""
import pytest

from extraction.fields import BY_NAME, fields_in
from extraction.service import ExtractionService
from extraction.trace import ExtractionTrace
from schemas import SearchFilters
from services.llm_service import LLMRateLimitError
from tests.conftest import make_pdf
from tests.test_extraction import NeedleLLM
from tests.test_extraction_pipeline import ADD2, BASE, FIRST, LATEST, BudgetLLM, ListRetrieval, sr

FILLER = "General terms and conditions apply to all purchases made under this solicitation. " * 6


@pytest.fixture
def harbor(engine, settings):
    """A bid whose key facts sit in passages the old evidence selection tended to miss: the named
    buyer only on the cover page, a required form only in a forms section, tiers on several pages."""
    d = settings.bids_root / "HarborSchools"
    d.mkdir()
    make_pdf(d / "IFB 9-77 Devices.pdf", [
        "INVITATION FOR BID 9-77 Classroom Devices\nBuyer LEE, PAT\nEmail PLEE@harbor.example",
        "PROCUREMENT SERVICES\nGeneral questions: Procurement Services, 555-010-2000, help@harbor.example\n" + FILLER,
        "FORMS\nFORM 77 - CERTIFICATE OF INTERESTED PARTIES must be completed and submitted with the response.\n"
        "A W-9 and a certificate of insurance are required after award.",
        "Tier 1 Teacher Laptop Minimum Requirements\nIntel i5, 16GB RAM, 512GB SSD, 14 inch display, 3-year warranty",
        "Tier 2 Student Chromebook Minimum Requirements\nCeleron N100, 8GB RAM, 64GB eMMC, 11.6 inch display",
        "Tier 3 Student Windows Laptop Minimum Requirements\nIntel i3, 8GB RAM, 256GB SSD, 14 inch display",
        "PRICE LINES\nQuantity of devices: Tier 1 Teacher Laptop 200 units; Tier 2 Student Chromebook 3000 units; "
        "Tier 3 Student Windows Laptop 1500 units",
    ])
    engine.indexing.index_folder(d)
    return engine


def texts(ev):
    return "\n".join(p.text for p in ev.passages)


# ----------------------------------------------------------------------- evidence selection
def test_interleaving_keeps_each_querys_best_hit_under_the_cap():
    a = [sr(i, f"alpha {i}", page=i) for i in range(1, 6)]
    b = sr(9, "beta best", page=9)

    class TwoQueries(ListRetrieval):
        def search(self, query, top_k=5, filters=None, mode=None):
            pool = [b] if query.startswith("SKU list") else a  # Part_no's 2nd query finds `b` at rank 1
            return [p.model_copy(update={"rank": i}) for i, p in enumerate(pool[:top_k], start=1)]

    svc = ExtractionService(TwoQueries([]), NeedleLLM(), per_query=5, max_passages=2)
    ev = svc.gather_evidence([BY_NAME["Part_no"]], "Bid")
    assert [p.text for p in ev.passages] == ["alpha 1", "beta best"]  # rank-1 of each query first


def test_multi_value_fields_get_a_larger_share():
    single = [sr(i, f"single {i}", page=i) for i in range(1, 9)]
    multi = [sr(20 + i, f"multi {i}", page=20 + i) for i in range(1, 9)]

    class ByField(ListRetrieval):
        def search(self, query, top_k=5, filters=None, mode=None):
            pool = single if "payment" in query else multi  # Payment Terms vs. documentation queries
            return [p.model_copy(update={"rank": i}) for i, p in enumerate(pool[:top_k], start=1)]

    svc = ExtractionService(ByField([]), NeedleLLM(), per_query=8, max_passages=6)
    ev = svc.gather_evidence([BY_NAME["Payment Terms"], BY_NAME["Any Additional Documentation Required"]], "Bid")
    got = [p.text.split()[0] for p in ev.passages]
    assert got.count("multi") == 4 and got.count("single") == 2


def test_named_buyer_required_form_and_all_tiers_reach_the_model(harbor):
    svc = ExtractionService(harbor.retrieval, NeedleLLM(), mode="hybrid")
    identity = texts(svc.gather_evidence(fields_in("identity"), "HarborSchools"))
    assert "LEE, PAT" in identity and "PLEE@harbor.example" in identity
    commercial = texts(svc.gather_evidence(fields_in("commercial_legal"), "HarborSchools"))
    assert "FORM 77" in commercial
    product = texts(svc.gather_evidence(fields_in("product"), "HarborSchools"))
    assert all(t in product for t in ("Tier 1 Teacher", "Tier 2 Student", "Tier 3 Student Windows"))


def test_named_buyer_extracted_and_verified(harbor):
    llm = NeedleLLM({"contact_info": ("LEE, PAT, PLEE@harbor.example", ["Buyer LEE, PAT"]),
                     "Any Additional Documentation Required": (
                         ["FORM 77 - Certificate of Interested Parties", "W-9", "Certificate of insurance"],
                         ["FORM 77", "W-9"]),
                     "Product": (["Tier 1 Teacher Laptop (200)", "Tier 2 Student Chromebook (3000)",
                                  "Tier 3 Student Windows Laptop (1500)"], ["PRICE LINES"])})
    rec = ExtractionService(harbor.retrieval, llm, mode="hybrid", reconcile_addendums=False).extract_bid("HarborSchools")
    for name in ("contact_info", "Any Additional Documentation Required", "Product"):
        assert rec.fields[name].status == "verified", (name, rec.fields[name].notes)
    assert rec.fields["contact_info"].sources[0].page == 1  # the cover page with the named buyer


def test_field_hints_are_in_the_prompt():
    from extraction.service import build_extraction_messages

    msg = build_extraction_messages(fields_in("identity"), [sr(1, "x")])[1]["content"]
    assert "Prefer the named buyer" in msg and "Procurement contact: name, email, phone." in msg


# ----------------------------------------------------------------------- reconciliation robustness
ADD_DOCS = sr(4, "ADDENDUM No. 2\nForm 88 is added: vendors must now submit Form 88 with the response.",
              doc_type="addendum", file="Addendum 2.pdf", add=2, page=2)
BASE_DOCS = sr(5, "REQUIRED DOCUMENTS\nSubmit a W-9 with the response.", file="RFP-9.pdf", page=7)


def run(llm, passages, **kw):
    trace = ExtractionTrace("Bid")
    rec = ExtractionService(ListRetrieval(list(passages)), llm, **kw).extract_bid("Bid", trace)
    return rec, trace


def test_reconciliation_is_requested_per_field_group():
    llm = NeedleLLM({"Due Date": FIRST}, scoped={"addendum": {"Due Date": LATEST}, "original": {"Due Date": FIRST}})
    rec, trace = run(llm, (BASE, ADD2))
    addendum_calls = [c["keys"] for c in llm.calls if c["scope"] == "addendum"]
    assert len(addendum_calls) == 3 and all(len(k) <= 4 for k in addendum_calls)  # groups, not one 10-field call
    assert [c.field for c in rec.addendum_changes] == ["Due Date"]


def test_large_reconciliation_outputs_are_split_and_recovered():
    # only 2 fields fit in the output budget: per-group requests of 4 fields are split once
    llm = BudgetLLM({"Due Date": FIRST}, max_fields=2,
                    scoped={"addendum": {"Due Date": LATEST}, "original": {"Due Date": FIRST}})
    rec, trace = run(llm, (BASE, ADD2))
    assert [c.field for c in rec.addendum_changes] == ["Due Date"]
    splits = {e["call"] for e in trace.events if e["stage"] == "split"}
    assert "reconcile/addendum/dates_logistics" in splits
    assert not any(e.get("outcome") == "no_validated_addendum_value" and e.get("addendum_status") == "error"
                   for e in trace.events if e["stage"] == "reconcile")


def test_partial_reconciliation_failure_keeps_other_groups():
    class FailProduct(NeedleLLM):
        def complete(self, messages, model=None):
            user = messages[1]["content"]
            if "come from ADDENDUMS only" in user and '"Model_no"' in user:
                self.calls.append({"keys": [], "passages": {}, "scope": "addendum"})
                raise LLMRateLimitError("rate limited")
            return super().complete(messages, model)

    llm = FailProduct({"Due Date": FIRST, "Any Additional Documentation Required": (["W-9"], ["Submit a W-9"])},
                      scoped={"addendum": {"Due Date": LATEST,
                                           "Any Additional Documentation Required": (["Form 88"], ["Form 88 is added"])},
                              "original": {"Due Date": FIRST}})
    rec, trace = run(llm, (BASE, ADD2, ADD_DOCS, BASE_DOCS), max_retries=0)
    assert {c.field for c in rec.addendum_changes} == {"Due Date", "Any Additional Documentation Required"}
    docs = next(c for c in rec.addendum_changes if c.field == "Any Additional Documentation Required")
    assert docs.previous_value == ["W-9"] and docs.value == ["Form 88"] and docs.addendum_number == 2
    failed = [e for e in trace.events if e["stage"] == "reconcile" and e.get("field") == "Product"]
    assert failed and failed[0]["outcome"] == "no_validated_addendum_value"
    assert rec.fields["Product Specification"].status != "error"  # main extraction result untouched
