"""Evidence passages reach the prompts intact or as a bounded, relevant excerpt - never a
blind head cut that drops table rows after character 1,200."""
import re

from extraction.fields import BY_NAME, fields_in
from extraction.service import build_extraction_messages
from schemas import SearchResult
from services.qa_service import EVIDENCE_PASSAGE_CHARS, build_messages, excerpt_passage, format_evidence


def sr(i, text, file="PORFP.pdf", page=3):
    return SearchResult(rank=i, score=1.0, chunk_id=f"Bid:{file}:{i}", text=text, bid_id="Bid", file_name=file,
                        doc_type="rfp", page_number=page, page_end=page, source_path=f"/{file}",
                        citation=f"{file}, p. {page}")


FILLER = "\n".join(f"| Section {i} | General terms and conditions apply to this purchase order {i} |"
                   for i in range(1, 19))
LATE_ROWS = ("| 1. SI# CC7802 Dell Latitude 5550 laptop | SI# CC7802 | 30 |\n"
             "| 2. Dell Thunderbolt 4 Dock - WD22TB4 | WD22TB4 | 30 |\n"
             "| Delivery | Delivery shall be made within 45 days of award |\n"
             "| Invoices | Email invoices within 10 days of delivery with proof of delivery |\n"
             "| 1. Dell Limited Hardware Warranty Extended for all machines purchased - 3 Years |")
# the shape of the real Bid2 PORFP p.3 chunk: ~1,970 chars, the facts start after char 1,200
TABLE = "PURCHASE ORDER REQUEST FOR PROPOSALS > Items\n" + FILLER + "\n" + LATE_ROWS
FACTS = ["SI# CC7802 | 30", "WD22TB4 | 30", "within 45 days of award", "within 10 days of delivery",
         "Warranty Extended for all machines purchased - 3 Years"]


def test_fixture_puts_the_facts_after_char_1200():
    assert 1200 < len(TABLE) <= EVIDENCE_PASSAGE_CHARS
    assert all(TABLE.index(f) > 1200 for f in FACTS)


def test_table_chunk_reaches_extraction_prompt_whole():
    for group in ("dates_logistics", "commercial_legal", "product"):
        user = build_extraction_messages(fields_in(group), [sr(1, TABLE)])[-1]["content"]
        assert all(f in user for f in FACTS), group
        assert "(bid: Bid; file: PORFP.pdf; page: 3; type: rfp)" in user


def test_long_passage_keeps_relevant_rows_heading_and_order():
    long_table = TABLE.replace(FILLER, FILLER + "\n" + FILLER + "\n" + FILLER)  # ~4,400 chars
    assert len(long_table) > EVIDENCE_PASSAGE_CHARS
    user = build_extraction_messages([BY_NAME["Delivery Date"], BY_NAME["Payment Terms"]],
                                     [sr(1, long_table)])[-1]["content"]
    body = user.split("Evidence passages:", 1)[1]
    assert "within 45 days of award" in body and "within 10 days of delivery" in body
    assert "PURCHASE ORDER REQUEST FOR PROPOSALS > Items" in body  # section heading kept
    assert "…" in body  # skipped rows are marked
    assert body.index("45 days") < body.index("10 days")  # document order kept
    kept_rows = [ln for ln in body.splitlines() if ln.startswith("|")]
    assert all(ln in long_table for ln in kept_rows)  # whole rows, never cut mid-row


def test_excerpt_is_bounded_and_identity_when_it_fits():
    assert excerpt_passage(TABLE, len(TABLE)) == TABLE
    long_text = TABLE * 4
    out = excerpt_passage(long_text, 1500, [re.compile("WD22TB4")])
    assert len(out) <= 1500 + 40 and "WD22TB4" in out


def test_single_huge_line_is_windowed_around_the_match():
    line = "x " * 3000 + "Delivery within 45 days of award " + "y " * 3000
    out = excerpt_passage(line, 1000, [re.compile("delivery", re.I)])
    assert "Delivery within 45 days of award" in out and len(out) < 1100


def test_total_budget_shrinks_each_passage_but_keeps_metadata():
    for n, share in ((14, 30000 // 14), (40, 900)):  # 900: floor below which an excerpt is useless
        passages = [sr(i, TABLE * 2, file=f"Doc{i}.pdf", page=i) for i in range(1, n + 1)]
        blocks = format_evidence(passages, total_chars=30000).split("\n\n")
        assert len(blocks) == n
        for i, b in enumerate(blocks, start=1):
            head, body = b.split("\n", 1)
            assert head == f"[{i}] (bid: Bid; file: Doc{i}.pdf; page: {i}; type: rfp)"
            assert len(body) <= share + 10


def test_ask_prompt_keeps_question_relevant_row_of_a_long_passage():
    long_table = TABLE.replace(FILLER, FILLER + "\n" + FILLER + "\n" + FILLER)
    user = build_messages("What is the delivery deadline after award?", [sr(1, long_table)])[-1]["content"]
    assert "within 45 days of award" in user
