import pytest

from ingestion.metadata import (
    classify_doc_type,
    extract_addendum_number,
    extract_document_date,
    make_doc_id,
    parse_date,
    pdf_metadata_date,
)
from schemas import DocType


@pytest.mark.parametrize(
    "name,head,expected",
    [
        ("Bid Information.html", "", DocType.BID_PAGE),
        ("Addendum 2 RFP JA-207652 Devices.pdf", "", DocType.ADDENDUM),
        ("notice.pdf", "ADDENDUM No. 3 RFP 123", DocType.ADDENDUM),
        ("Contract_Affidavit.pdf", "", DocType.AFFIDAVIT),
        ("Mercury_Affidavit.pdf", "", DocType.AFFIDAVIT),
        ("Dell_Laptop_Specs.pdf", "", DocType.SPECS),
        ("PORFP_-_Dell_Laptop_Final.pdf", "", DocType.RFP),
        ("JA-207652 Devices FINAL.pdf", "Dallas ISD Request For Proposal 168884", DocType.RFP),
        ("random.pdf", "lorem ipsum", DocType.OTHER),
    ],
)
def test_classify_doc_type(name, head, expected):
    assert classify_doc_type(name, head) == expected


def test_addendum_body_mention_does_not_make_rfp_an_addendum():
    head = "Request For Proposal 168884. " + "x " * 300 + "responses will be provided by means of an addendum"
    assert classify_doc_type("JA-207652 FINAL.pdf", head) == DocType.RFP


@pytest.mark.parametrize(
    "name,head,expected",
    [
        ("Addendum 2 RFP.pdf", "", 2),
        ("Addendum_No_10.pdf", "", 10),
        ("notice.pdf", "ADDENDUM No. 3", 3),
        ("Amendment #4.pdf", "", 4),
        ("Addendum Two.pdf", "", 2),
        ("RFP.pdf", "no number here", None),
    ],
)
def test_extract_addendum_number(name, head, expected):
    assert extract_addendum_number(name, head) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("due 06/10/2024 at 2pm", "2024-06-10"),
        ("26-MAY-2024 08:00:00", "2024-05-26"),
        ("July 9, 2024 at 2:00 PM", "2024-07-09"),
        ("Sept 3 2024", "2024-09-03"),
        ("2024-05-24", "2024-05-24"),
        ("no date", None),
        ("13/45/2024", None),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_extract_document_date_uses_label():
    head = "PORFP Number: #E20 PORFP Issue Date: mm/dd/yyyy 05/24/2024 PROPOSAL DUE DATE 06/10/2024"
    assert extract_document_date(head) == "2024-05-24"
    assert extract_document_date("Closing 06/10/2024 only") is None


def test_pdf_metadata_date():
    assert pdf_metadata_date("D:20240613175856-05'00'") == "2024-06-13"
    assert pdf_metadata_date(None) is None
    assert pdf_metadata_date("garbage") is None


def test_doc_id_is_deterministic_and_bid_scoped():
    assert make_doc_id("Bid1", "abc") == make_doc_id("Bid1", "abc")
    assert make_doc_id("Bid1", "abc") != make_doc_id("Bid2", "abc")
