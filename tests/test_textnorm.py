"""Unicode normalisation and negation-aware mention detection (extraction.textnorm), and the
completeness check (`coverage_gaps`) that uses them."""
import pytest

from extraction.fields import BY_NAME
from extraction.schemas import FieldResult
from extraction.service import coverage_gaps
from extraction.textnorm import AMBIGUOUS, MISSING, NEGATED, POSITIVE, mention_state, normalize_text
from schemas import SearchResult

WARRANTY = r"\bwarrant(y|ies)\b"


@pytest.mark.parametrize("raw", ["16 GB", "16 GB", "16 GB", "16 GB", "16　GB",
                                 "16 ​GB", "16  GB", "１６ GB"])
def test_unicode_spaces_fold_to_one_ascii_space(raw):
    assert normalize_text(raw) == "16 gb"


def test_dash_variants_fold_to_hyphen():
    for d in "‐‑‒–—−":
        assert normalize_text(f"JA{d}207652") == "ja-207652"


@pytest.mark.parametrize("text,state", [
    ("Dell ProSupport 3-year onsite warranty", POSITIVE),
    ("Warranty: 3 years parts and labour", POSITIVE),
    ("Dell Limited Hardware Warranty Extended - 3 Years", POSITIVE),
    ("warranty duration and coverage not found in provided passages", NEGATED),
    ("Warranty not stated.", NEGATED),
    ("No warranty is offered.", NEGATED),
    ("The quote does not include any warranty", NEGATED),
    ("warranty: none", NEGATED),
    ("Warranty: N/A", NEGATED),
    ("Warranty TBD", AMBIGUOUS),
    ("Warranty terms to be determined by the agency", AMBIGUOUS),
    ("Extended warranty possibly available", AMBIGUOUS),
    ("16 GB RAM, 256 GB SSD", MISSING),
    ("", MISSING),
])
def test_mention_state(text, state):
    assert mention_state(WARRANTY, text) == state


def test_one_positive_mention_wins_over_a_negated_one():
    assert mention_state(WARRANTY, "Accidental damage not covered by warranty; 3-year warranty included") == POSITIVE
    assert mention_state(WARRANTY, "No extended warranty. Standard 1-year warranty applies.") == POSITIVE


def test_permission_wording_and_number_abbreviation_are_not_negation_or_hedge():
    assert mention_state(WARRANTY, "Bidders may offer warranty upgrades") == POSITIVE
    assert mention_state(r"z-100", "Bid No Z-100") == POSITIVE
    assert mention_state(r"manual", "Electronic (PDF not required) or manual submission") == POSITIVE


# ----------------------------------------------------------------------- completeness check
def sr(i, text, file="RFP.pdf"):
    return SearchResult(rank=i, score=1.0, chunk_id=f"Bid:{file}:{i}", text=text, bid_id="Bid", file_name=file,
                        doc_type="rfp", page_number=1, page_end=1, source_path=f"/{file}", citation=f"{file}, p. 1")


SPEC = BY_NAME["Product Specification"]
STATED = sr(1, "Item 1 Laptop | 16 GB | 256 GB SSD\nDell Limited Hardware Warranty Extended - 3 Years")


def test_negated_value_does_not_cover_a_stated_warranty():
    gaps, support = coverage_gaps(SPEC, "16 GB, 256 GB; warranty not found in provided passages", [STATED], {})
    assert gaps == ["warranty"] and support == [STATED]


def test_positive_value_covers_the_warranty():
    assert coverage_gaps(SPEC, "16 GB, 256 GB; 3-year extended hardware warranty", [STATED], {}) == ([], [])


def test_ambiguous_value_does_not_cover_it():
    assert coverage_gaps(SPEC, "16 GB, 256 GB; warranty TBD", [STATED], {})[0] == ["warranty"]


def test_negated_or_missing_evidence_is_no_gap():
    no_warranty = sr(2, "Laptop 16 GB 256 GB. No warranty is offered with this quote.")
    assert coverage_gaps(SPEC, "16 GB, 256 GB SSD", [no_warranty], {}) == ([], [])
    assert coverage_gaps(SPEC, "16 GB, 256 GB SSD", [sr(3, "Laptop 16 GB 256 GB")], {}) == ([], [])


def test_model_check_is_unicode_normalised():
    models = {"Model_no": FieldResult(status="verified", value=["WD22TB4"])}
    dock = sr(4, "2. Dell Thunderbolt 4 Dock - WD22TB4 | qty 30")
    product = BY_NAME["Product"]
    assert coverage_gaps(product, "Dell Thunderbolt 4 Dock WD22TB4 (30)", [dock], models) == ([], [])
    assert coverage_gaps(product, "Dell Latitude 5550 (30)", [dock], models)[0] == ["model WD22TB4"]
