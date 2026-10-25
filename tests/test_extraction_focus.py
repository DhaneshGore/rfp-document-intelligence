"""Focused single-field retries (offline, stub LLMs, invented fixtures).

* not_found although a passage addresses the field (its cue wording) -> one focused retry
  (Delivery Date, Payment Terms); preserved when the evidence does not support a value;
* verified but incomplete (coverage gaps): Product must name every verified model incl.
  accessories, Product Specification must include a warranty the evidence states.
"""
import json

import pytest

from agents import Orchestrator
from extraction.fields import BY_NAME, GROUPS
from extraction.schemas import FieldResult
from extraction.service import ExtractionService, coverage_gaps, focused_retry_plan
from extraction.trace import ExtractionTrace
from tests.test_extraction import NeedleLLM
from tests.test_extraction_pipeline import ListRetrieval, sr

FOCUS_NOTES = ("These passages may state this field", "Make the value complete", "A previous answer used a seller")

INVOICING = sr(1, "Invoicing Instructions: | 9. Delivery within 45 days of Award. 1. Email invoices to: "
                  "ap@agency.example 2. Invoice(s) shall be submitted within 10 days of delivering the equipment "
                  "and include the PO number.", file="PO-RFP.pdf", page=2)
ADDRESS = sr(2, "Delivery Address: 80 Harbor Street, Room 9. Agency on-site contact: J. Doe.", file="PO-RFP.pdf", page=3)
TABLE = sr(3, "Product Name | Product Description | Model # | Qty\n1. Laptop X1 | Laptop X1 business notebook | "
              "X1-15 | 30\n2. Docking Station D9 | USB-C docking station | D9-USB | 30\nWarranty Requirements | "
              "Manufacturer extended hardware warranty - 3 Years from the date of delivery",
           file="PO-RFP.pdf", page=4)
QUOTE = sr(4, "Laptop X1 configuration | SKU\nIntel Core i5 processor | 111-AAA\n16 GB DDR5 memory | 222-BBB\n"
              "512 GB SSD storage | 333-CCC\n14 inch FHD display | 444-DDD", file="Quote.pdf", page=1)


class FocusLLM(NeedleLLM):
    """Group calls answer from `answers`; focused single-field calls (recognised by the
    orchestration note in the prompt) answer from `focus_answers`."""

    def __init__(self, answers=None, focus_answers=None, **kw):
        super().__init__(answers, **kw)
        self.group_answers, self.focus_answers = dict(answers or {}), dict(focus_answers or {})

    def complete(self, messages, model=None):
        focused = any(n in messages[1]["content"] for n in FOCUS_NOTES)
        self.answers = {**self.group_answers, **self.focus_answers} if focused else self.group_answers
        resp = super().complete(messages, model)
        self.calls[-1]["focused"] = focused
        return resp


def run(llm, passages, trace=None, **kw):
    kw.setdefault("reconcile_addendums", False)
    kw.setdefault("max_retries", 0)
    return ExtractionService(ListRetrieval(list(passages)), llm, **kw).extract_bid("Bid", trace or ExtractionTrace("Bid"))


def focused_calls(llm):
    return [c["keys"][0] for c in llm.calls if c.get("focused")]


# ----------------------------------------------------------------------- missed fields
def test_missed_delivery_and_payment_recovered_by_focused_retry():
    llm = FocusLLM(focus_answers={
        "Delivery Date": ("Delivery within 45 days of Award", ["Delivery within 45 days"]),
        "Payment Terms": ("Email invoices to ap@agency.example within 10 days of delivering the equipment, with the PO "
                          "number", ["Invoice(s) shall be submitted within 10 days"]),
    })
    trace = ExtractionTrace("Bid")
    rec = run(llm, (INVOICING, ADDRESS), trace)
    for name in ("Delivery Date", "Payment Terms"):
        f = rec.fields[name]
        assert f.status == "verified" and f.attempts == 2, (name, f.notes)
        assert [(s.file, s.page) for s in f.sources] == [("PO-RFP.pdf", 2)]
    # only fields whose evidence addresses them are retried (not every not_found field)
    assert set(focused_calls(llm)) <= {"Delivery Date", "Payment Terms", "contact_info", "Bid Submission Type"}
    assert {"Delivery Date", "Payment Terms"} <= set(focused_calls(llm))
    assert "Term of Bid" not in focused_calls(llm) and "Bid Bond Requirement" not in focused_calls(llm)
    assert all(len(c["keys"]) == 1 for c in llm.calls if c.get("focused"))  # single field per call
    events = [e for e in trace.events if e["stage"] == "focus" and e.get("outcome") == "accepted"]
    assert {e["field"] for e in events} >= {"Delivery Date", "Payment Terms"}


def test_focused_retry_gets_only_the_fields_supporting_evidence():
    llm = FocusLLM(focus_answers={"Delivery Date": ("Delivery within 45 days of Award", ["Delivery within 45 days"])})
    run(llm, (INVOICING, ADDRESS, QUOTE))
    call = next(c for c in llm.calls if c.get("focused") and c["keys"] == ["Delivery Date"])
    assert len(call["passages"]) == 1 and "Delivery within 45 days" in call["passages"][1]  # not ADDRESS / QUOTE


def test_not_found_preserved_when_evidence_does_not_support_a_value():
    llm = FocusLLM(focus_answers={})  # the focused answer is still null
    rec = run(llm, (INVOICING,))
    f = rec.fields["Delivery Date"]
    assert f.status == "not_found" and f.value is None and "Focused retry" in f.notes and "kept" in f.notes


def test_ungrounded_focused_answer_is_rejected():
    llm = FocusLLM(focus_answers={"Delivery Date": ("Delivery within 30 days of Award", ["Delivery within 45 days"])})
    rec = run(llm, (INVOICING,))
    assert rec.fields["Delivery Date"].status == "not_found"  # 30 is not in the cited passage


def test_no_retry_without_passages_that_address_the_field():
    llm = FocusLLM()
    rec = run(llm, (ADDRESS,))  # "Delivery Address" is not delivery timing
    assert "Delivery Date" not in focused_calls(llm) and "Payment Terms" not in focused_calls(llm)
    assert rec.fields["Delivery Date"].status == "not_found"


def test_focused_calls_are_bounded():
    llm = FocusLLM()
    run(llm, (INVOICING, ADDRESS, TABLE, QUOTE), max_focus_calls=2)
    assert len(focused_calls(llm)) <= 2


# ----------------------------------------------------------------------- payment terms vs. subcontractor clauses
MWBE_PAY = sr(5, "M/WBE Pay Activity Report\n- All Prime vendors must pay all submitted invoices, including retainage "
                 "to subcontractors, suppliers, or entities within 10 days of receiving payment from the district;\n"
                 "- No Prime vendor may withhold retainage greater than 5% from the subcontractor.",
              file="RFP-MWBE.pdf", page=51)
MWBE_CONTRACT = sr(6, "Prime vendor agrees to establish a written contract with each subcontractor. At minimum, the "
                      "contract should include the scope of work, payment terms, prompt payment clause and retainage "
                      "clause.", file="RFP-MWBE.pdf", page=51)
FINANCING = sr(7, "If Customer fails to enter into such Financing Agreement within 120 days after shipment, Customer "
                  "shall promptly pay the Supplier invoice amounts directly to Supplier.", file="Quote.pdf", page=3)
AGENCY_PAY = sr(8, "PAYMENT\nThe District pays Net 30 after receipt of a correct invoice. Invoices must reference the "
                   "PO number.", file="RFP-MWBE.pdf", page=12)
SUBCONTRACTOR_ANSWER = ("Prime vendors must pay submitted invoices, including retainage, to subcontractors within 10 days "
                        "of receiving payment from the district", ["All Prime vendors must pay"])


def test_subcontractor_payment_clauses_do_not_trigger_a_focused_retry():
    llm = FocusLLM(focus_answers={"Payment Terms": SUBCONTRACTOR_ANSWER})
    rec = run(llm, (MWBE_PAY, MWBE_CONTRACT, FINANCING))
    assert "Payment Terms" not in focused_calls(llm)
    p = rec.fields["Payment Terms"]
    assert p.status == "not_found" and p.value is None


def test_subcontractor_payment_value_is_rejected_even_from_the_group_call():
    llm = FocusLLM({"Payment Terms": SUBCONTRACTOR_ANSWER})  # the model misreads the M/WBE clause
    rec = run(llm, (MWBE_PAY, MWBE_CONTRACT))
    p = rec.fields["Payment Terms"]
    assert p.status == "not_found" and p.value is None and p.sources == []
    assert "describes a different clause" in p.notes


def test_agency_payment_terms_next_to_subcontractor_clauses_are_kept():
    llm = FocusLLM(focus_answers={"Payment Terms": ("Net 30 after receipt of a correct invoice; invoices must "
                                                    "reference the PO number", ["pays Net 30"])})
    trace = ExtractionTrace("Bid")
    rec = run(llm, (MWBE_PAY, AGENCY_PAY, MWBE_CONTRACT), trace)
    p = rec.fields["Payment Terms"]
    assert p.status == "verified" and "Net 30" in p.value and [s.page for s in p.sources] == [12]
    focus = next(e for e in trace.events if e["stage"] == "focus" and e.get("reason"))
    assert focus["passages"] == [AGENCY_PAY.chunk_id]  # subcontractor passages not sent to the focused call


def test_mixed_value_keeps_agency_terms():
    from extraction.service import describes_something_else

    spec = BY_NAME["Payment Terms"]
    assert describes_something_else(spec, SUBCONTRACTOR_ANSWER[0])
    assert describes_something_else(spec, ["Customer shall pay the Supplier invoice under the Financing Agreement"])
    assert not describes_something_else(spec, "Net 30 after receipt of invoice. Prime vendors pay subcontractors "
                                              "within 10 days.")
    assert not describes_something_else(spec, "Email invoices within 10 days of delivery")


def test_payment_prompt_states_agency_terms_only():
    from extraction.service import build_extraction_messages

    msg = build_extraction_messages([BY_NAME["Payment Terms"]], [AGENCY_PAY])[1]["content"]
    assert "agency" in msg and "subcontractors" in msg and "return null" in msg


# ----------------------------------------------------------------------- solicitation over seller documents
QUOTE_TERMS = sr(9, "Terms of Sale\nUnless otherwise stated herein, pricing is valid for thirty days from the date of "
                    "this Quote. Taxes and freight are estimates; the final amounts shall be stated on the relevant "
                    "invoice.", doc_type="specs", file="Vendor-Quote.pdf", page=3)
AGENCY_INV = sr(10, "Invoicing Instructions: 1. Email invoices to: ap@agency.example 2. Invoices shall be submitted "
                    "within 10 days of delivering the equipment and include the PO number.",
                file="PO-RFP.pdf", page=2)
QUOTE_ANSWER = ("The final amounts shall be stated on the relevant invoice", ["final amounts shall be stated"])
AGENCY_ANSWER = ("Email invoices to ap@agency.example within 10 days of delivering the equipment, with the PO number",
                 ["Email invoices to"])


def test_agency_instructions_preferred_over_higher_ranked_quote_wording():
    llm = FocusLLM(focus_answers={"Payment Terms": AGENCY_ANSWER})
    trace = ExtractionTrace("Bid")
    rec = run(llm, (QUOTE_TERMS, AGENCY_INV), trace)  # the quote ranks first
    call = next(c for c in llm.calls if c.get("focused") and c["keys"] == ["Payment Terms"])
    assert len(call["passages"]) == 1 and "Email invoices" in call["passages"][1]  # the quote is not sent
    p = rec.fields["Payment Terms"]
    assert p.status == "verified" and [(s.file, s.doc_type) for s in p.sources] == [("PO-RFP.pdf", "rfp")]


def test_value_cited_only_from_a_quote_is_re_asked_from_the_solicitation():
    llm = FocusLLM({"Payment Terms": QUOTE_ANSWER}, focus_answers={"Payment Terms": AGENCY_ANSWER})
    trace = ExtractionTrace("Bid")
    rec = run(llm, (QUOTE_TERMS, AGENCY_INV), trace)
    p = rec.fields["Payment Terms"]
    assert p.status == "verified" and "Email invoices" in p.value and {s.doc_type for s in p.sources} == {"rfp"}
    assert any(e["stage"] == "focus" and e.get("reason") == "preferred_source" for e in trace.events)


def test_quote_value_downgraded_when_the_solicitation_retry_fails():
    llm = FocusLLM({"Payment Terms": QUOTE_ANSWER}, focus_answers={})  # focused answer: null
    rec = run(llm, (QUOTE_TERMS, AGENCY_INV))
    p = rec.fields["Payment Terms"]
    assert p.status == "unverified" and p.confidence <= 0.5 and "seller documents" in p.notes


def test_quote_value_kept_when_the_solicitation_says_nothing_about_payment():
    llm = FocusLLM({"Payment Terms": QUOTE_ANSWER})
    rec = run(llm, (QUOTE_TERMS, ADDRESS))  # no solicitation passage addresses payment
    assert "Payment Terms" not in focused_calls(llm)
    assert rec.fields["Payment Terms"].status == "verified"  # the only evidence there is


# ----------------------------------------------------------------------- mixed agency + subcontractor values
MIXED = ("The District pays Net 30 after receipt of a correct invoice. All Prime vendors must pay all submitted invoices, "
         "including retainage to subcontractors, within 10 days of receiving payment from the district.")


def test_mixed_value_reduced_to_supported_agency_terms():
    llm = FocusLLM({"Payment Terms": (MIXED, ["pays Net 30", "All Prime vendors must pay"])})
    rec = run(llm, (AGENCY_PAY, MWBE_PAY))
    p = rec.fields["Payment Terms"]
    assert p.status == "verified", p.notes
    assert p.value == "The District pays Net 30 after receipt of a correct invoice."
    assert "subcontractor" not in p.value.lower() and "retainage" not in p.value.lower()
    assert [s.page for s in p.sources] == [12]  # the M/WBE citation was dropped with its clause
    assert "removed unrelated clause" in p.notes


def test_mixed_list_value_keeps_only_agency_items():
    llm = FocusLLM({"Payment Terms": (["Net 30 after receipt of a correct invoice",
                                       "Prime vendors pay subcontractors within 10 days, including retainage"],
                                      ["pays Net 30", "All Prime vendors must pay"])})
    p = run(llm, (AGENCY_PAY, MWBE_PAY)).fields["Payment Terms"]
    assert p.status == "verified" and p.value == ["Net 30 after receipt of a correct invoice"]


def test_mixed_value_with_unsupported_agency_part_is_not_verified():
    llm = FocusLLM({"Payment Terms": ("The District pays Net 45 after receipt of invoice. Prime vendors must pay "
                                      "subcontractors within 10 days.", ["pays Net 30", "All Prime vendors must pay"])})
    p = run(llm, (AGENCY_PAY, MWBE_PAY)).fields["Payment Terms"]
    assert p.status != "verified"  # 45 is not in the cited agency passage -> rejected safely


# ----------------------------------------------------------------------- product completeness
MODELS = (["X1-15", "D9-USB"], ["X1-15 | 30"])


def test_product_line_items_and_accessories_completed_with_all_sources():
    llm = FocusLLM({"Model_no": MODELS, "Product": (["Laptop X1"], ["Laptop X1 configuration"])},
                   focus_answers={"Product": (["Laptop X1 (model X1-15, SKU 111-AAA), qty 30",
                                               "Docking Station D9 (model D9-USB), qty 30"],
                                              ["1. Laptop X1", "Laptop X1 configuration"])})
    rec = run(llm, (TABLE, QUOTE))
    p = rec.fields["Product"]
    assert p.status == "verified" and p.attempts == 2, p.notes
    assert any("D9-USB" in v and "30" in v for v in p.value) and any("X1-15" in v for v in p.value)
    assert {s.file for s in p.sources} == {"PO-RFP.pdf", "Quote.pdf"}  # every source used is cited


def test_product_retry_that_drops_items_is_rejected():
    llm = FocusLLM({"Model_no": MODELS, "Product": (["Laptop X1"], ["Laptop X1 configuration"])},
                   focus_answers={"Product": (["Docking Station D9 (model D9-USB), qty 30"], ["2. Docking Station"])})
    rec = run(llm, (TABLE, QUOTE))
    assert rec.fields["Product"].value == ["Laptop X1"] and "kept" in rec.fields["Product"].notes


def test_product_quantity_not_invented():
    llm = FocusLLM({"Model_no": MODELS, "Product": (["Laptop X1"], ["Laptop X1 configuration"])},
                   focus_answers={"Product": (["Laptop X1 (X1-15), qty 30", "Docking Station D9 (D9-USB), qty 50"],
                                              ["1. Laptop X1"])})
    rec = run(llm, (TABLE, QUOTE))
    assert rec.fields["Product"].value == ["Laptop X1"]  # 50 is not in the evidence -> rejected


def test_coverage_gap_lists_missing_models():
    fields = {"Model_no": FieldResult(value=["X1-15", "D9-USB"], status="verified")}
    gaps, support = coverage_gaps(BY_NAME["Product"], ["Laptop X1 (X1-15)"], [TABLE, QUOTE], fields)
    assert gaps == ["model D9-USB"] and [p.chunk_id for p in support] == [TABLE.chunk_id]


# ----------------------------------------------------------------------- specification warranty
SPEC = ("Laptop X1: Intel Core i5 processor, 16 GB DDR5 memory, 512 GB SSD storage, 14 inch FHD display",
        ["Intel Core i5 processor"])


def test_specification_warranty_completed_when_stated():
    llm = FocusLLM({"Product Specification": SPEC},
                   focus_answers={"Product Specification": (
                       SPEC[0] + "; warranty: manufacturer extended hardware warranty - 3 Years from the date of delivery",
                       ["Intel Core i5 processor", "extended hardware warranty"])})
    rec = run(llm, (QUOTE, TABLE))
    s = rec.fields["Product Specification"]
    assert s.status == "verified" and "3 Years" in s.value and s.attempts == 2, s.notes


def test_no_warranty_evidence_no_retry_and_nothing_invented():
    llm = FocusLLM({"Product Specification": SPEC},
                   focus_answers={"Product Specification": (SPEC[0] + "; 3-year warranty", ["Intel Core i5"])})
    rec = run(llm, (QUOTE,))  # no passage mentions a warranty
    assert "Product Specification" not in focused_calls(llm)
    assert rec.fields["Product Specification"].value == SPEC[0]


def test_plan_has_no_tasks_for_complete_values():
    fields = {"Product Specification": FieldResult(value=SPEC[0] + "; 3 Years warranty", status="verified"),
              "Delivery Date": FieldResult(value="within 45 days", status="verified")}
    plan = focused_retry_plan(fields, {"Product Specification": [TABLE], "Delivery Date": [INVOICING]}, {})
    assert plan == []


# ----------------------------------------------------------------------- multi-agent path
def test_orchestrator_runs_focused_retries_as_agent_handoffs():
    llm = FocusLLM(focus_answers={"Delivery Date": ("Delivery within 45 days of Award", ["Delivery within 45 days"])})
    trace = ExtractionTrace("Bid")
    state = Orchestrator(ListRetrieval([INVOICING, ADDRESS]), llm, reconcile=False, max_retries=0).run_extraction(
        "Bid", trace=trace)
    assert state.final.fields["Delivery Date"].status == "verified"
    assert state.step("focus:Delivery Date").status == "done"
    assert state.verdicts["Delivery Date"].decision == "accept"  # the critic accepts: source was shown
    assert state.llm_calls == len(llm.calls) == len(GROUPS) + len(focused_calls(llm))
