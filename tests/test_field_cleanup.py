"""Value / citation / status clean-up after the model answers (offline, stub LLMs, invented fixtures).

* Delivery Date: a bid submission due date mixed into the delivery value is removed, with the
  citation that only supported it;
* identifier fields: cited passages that contain none of the identifiers are dropped, passages
  that do contain them (e.g. a bid page's item list under an odd section heading) are kept;
* an uncited "None" is a not_found, and a retry's clean not_found replaces an earlier rejected
  empty answer instead of leaving the field "unverified" for review.
"""
from extraction.fields import BY_NAME
from extraction.schemas import FieldResult
from extraction.service import better_result, says_absent, strip_unrelated_parts
from extraction.trace import ExtractionTrace
from tests.test_extraction import NeedleLLM
from tests.test_extraction_focus import ADDRESS, INVOICING, TABLE, run
from tests.test_extraction_pipeline import sr

SCHEDULE = sr(5, "Product Name | Model # | Qty | Due Date mm/dd/yyyy\n1. Laptop X1 | X1-15 | 30 | 06/10/2024",
              file="PO-RFP.pdf", page=3)
BID_PAGE = sr(6, "BID-77 - Laptops > Contact Information\nJ. Doe\nDescription: 1. Laptop X1 X1-15 30 2. Docking "
                 "Station D9 D9-USB 30", file="Bid Information.html", page=None, doc_type="bid_page")


# ----------------------------------------------------------------------- Delivery Date vs. bid due date
def test_bid_due_date_is_removed_from_the_delivery_value():
    llm = NeedleLLM({"Delivery Date": ("Within 45 days of award; scope table shows due date 06/10/2024",
                                       ["45 days of Award", "Due Date mm/dd/yyyy"])})
    rec = run(llm, (INVOICING, SCHEDULE))
    d = rec.fields["Delivery Date"]
    assert d.status == "verified", d.notes
    assert d.value == "Within 45 days of award"
    assert [s.page for s in d.sources] == [2]  # the schedule passage only supported the due date
    assert "removed unrelated clause" in d.notes and "06/10/2024" in d.notes


def test_delivery_clauses_that_name_a_due_date_are_kept():
    spec = BY_NAME["Delivery Date"]
    for value in ("Delivery due date 07/01/2024", "Deliver by the due date of 07/01/2024",
                  "Within 45 days of award"):
        assert strip_unrelated_parts(spec, value) == (value, [])
    assert strip_unrelated_parts(spec, ["Within 45 days of award", "Proposals are due 06/10/2024"]) == \
        (["Within 45 days of award"], ["Proposals are due 06/10/2024"])


def test_value_that_is_only_the_bid_due_date_is_not_a_delivery_date():
    llm = NeedleLLM({"Delivery Date": ("Due date 06/10/2024", ["Due Date mm/dd/yyyy"])})
    d = run(llm, (SCHEDULE,)).fields["Delivery Date"]
    assert d.status == "not_found" and d.value is None and "different clause" in d.notes


# ----------------------------------------------------------------------- identifier citations
def test_identifier_citation_without_the_identifiers_is_dropped():
    llm = NeedleLLM({"Model_no": (["X1-15", "D9-USB"], ["X1-15", "Delivery Address"])})
    m = run(llm, (TABLE, ADDRESS)).fields["Model_no"]
    assert m.status == "verified" and [s.page for s in m.sources] == [4]


def test_bid_page_item_list_stays_cited_whatever_its_section_heading():
    llm = NeedleLLM({"Model_no": (["X1-15", "D9-USB"], ["X1-15 | 30", "Description: 1. Laptop X1"]),
                     "Product": (["Laptop X1 - Qty 30", "Docking Station D9 - Qty 30"],
                                 ["X1-15 | 30", "Description: 1. Laptop X1"])})
    rec = run(llm, (TABLE, BID_PAGE))
    for name in ("Model_no", "Product"):
        f = rec.fields[name]
        assert f.status == "verified", (name, f.notes)
        assert {s.file for s in f.sources} == {"PO-RFP.pdf", "Bid Information.html"}, name


# ----------------------------------------------------------------------- null answers
def test_uncited_none_is_not_found_not_a_rejected_value():
    llm = NeedleLLM(overrides={"Pre Bid Meeting": {"value": "None", "refs": [], "notes": "Not found in documents"}})
    p = run(llm, (INVOICING,)).fields["Pre Bid Meeting"]
    assert p.status == "not_found" and p.value is None and p.sources == []
    assert p.notes.startswith("Not found in documents") and "Rejected" not in p.notes


def test_cited_none_is_kept_as_a_value():
    meeting = sr(7, "There will be no pre-bid meeting for this solicitation.", file="PO-RFP.pdf", page=1)
    llm = NeedleLLM({"Pre Bid Meeting": ("None", ["no pre-bid meeting"])})
    p = run(llm, (meeting,)).fields["Pre Bid Meeting"]
    assert p.value == "None" and p.status != "not_found" and [s.page for s in p.sources] == [1]


def test_says_absent():
    assert says_absent("None") and says_absent("not specified.") and says_absent(["None", "Not listed"])
    assert not says_absent("None required - no bid bond") and not says_absent(["None", "X1-15"])
    assert not says_absent(None)


def test_clean_not_found_replaces_a_rejected_empty_answer():
    rejected = FieldResult(status="unverified", value=None, confidence=0.0, notes="Rejected: ... cites no valid")
    not_found = FieldResult(status="not_found", value=None, confidence=0.6, notes="Not found in documents")
    rejected_value = FieldResult(status="unverified", value="Mandatory, 06/01/2024", confidence=0.4)
    assert better_result(not_found, rejected) and not better_result(rejected, not_found)
    assert not better_result(not_found, rejected_value)  # a value to review still outranks "nothing"


def test_none_citing_an_unknown_passage_is_not_found_and_summary_counts_no_review():
    llm = NeedleLLM(overrides={"Pre Bid Meeting": {"value": "None", "refs": [99], "notes": "Not found"}})
    rec = run(llm, (INVOICING, ADDRESS), ExtractionTrace("Bid"))
    p = rec.fields["Pre Bid Meeting"]
    assert p.status == "not_found" and p.value is None, p.notes
    assert rec.validation.unverified == 0
