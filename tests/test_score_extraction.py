"""The offline extraction scorer: incomplete values, multi-source citations, nulls."""
from eval.score_extraction import score_field

PRODUCT = {"expect": "value", "all_of": [["Latitude 5550", "laptop"], ["WD22TB4", "Dock"], ["30"]],
           "sources": ["PORFP", "Dell_Laptop_Specs"]}


def got(value, *files):
    return {"value": value, "sources": [{"file": f, "page": 1} for f in files]}


def test_complete_product_with_multiple_sources_is_correct():
    s, why = score_field(PRODUCT, got(["Dell Latitude 5550 (30)", "Dell Thunderbolt Dock WD22TB4 (30)"],
                                      "Dell_Laptop_Specs.pdf", "PORFP_-_Dell_Laptop_Final.pdf"))
    assert s == 1.0 and why == "correct"


def test_one_valid_source_among_several_is_enough():
    s, _ = score_field(PRODUCT, got(["Latitude 5550 x30", "WD22TB4 dock x30"], "unrelated.pdf", "PORFP.pdf"))
    assert s == 1.0


def test_incomplete_product_scores_partial_or_zero_by_coverage():
    two_of_three, why = score_field(PRODUCT, got(["Dell Latitude 5550 laptop, qty 30"], "PORFP.pdf"))
    assert two_of_three == 0.5 and "WD22TB4" in why
    one_of_three, why = score_field(PRODUCT, got(["Dell Latitude 5550 XCTO Base"], "Dell_Laptop_Specs.pdf"))
    assert one_of_three == 0.0 and why.startswith("incorrect: missing")


def test_wrong_source_and_invented_values_score_zero():
    s, why = score_field(PRODUCT, got(["Latitude 5550 x30", "WD22TB4 x30"], "other-bid.pdf"))
    assert s == 0.0 and why.startswith("unsupported")
    s, why = score_field({"expect": "null"}, got("Net 30"))
    assert s == 0.0 and why.startswith("invented")
    assert score_field({"expect": "null"}, got(None))[0] == 1.0
    assert score_field(PRODUCT, got(None))[0] == 0.0


SPEC = {"expect": "value", "all_of": [["16 GB", "16GB"], ["256 GB", "256GB"], ["warranty", "3 Year", "3 years"]],
        "sources": ["Dell_Laptop_Specs"]}


def test_unicode_spaces_and_dashes_are_normalised():
    # narrow no-break (U+202F), no-break (U+00A0), thin (U+2009) spaces as models emit them
    for sp in (" ", " ", " ", "　"):
        s, why = score_field(SPEC, got(f"16{sp}GB DDR5, 256{sp}GB SSD; 3-year warranty", "Dell_Laptop_Specs.pdf"))
        assert s == 1.0 and why == "correct", repr(sp)
    dash = {"expect": "value", "all_of": [["JA-207652"]], "sources": ["RFP"]}
    assert score_field(dash, got("JA‑207652", "RFP.pdf"))[0] == 1.0  # non-breaking hyphen


def test_negated_term_earns_no_credit():
    value = "16 GB DDR5, 256 GB SSD; warranty duration and coverage not found in provided passages"
    s, why = score_field(SPEC, got(value, "Dell_Laptop_Specs.pdf"))
    assert s == 0.5 and why == "partial: missing ['warranty']"
    assert score_field(SPEC, got("16 GB, 256 GB, no warranty", "Dell_Laptop_Specs.pdf"))[1] == \
        "partial: missing ['warranty']"
    # hedged ("TBD") is not a positive statement either
    assert score_field(SPEC, got("16 GB, 256 GB, warranty TBD", "Dell_Laptop_Specs.pdf"))[0] == 0.5


def test_negation_in_one_clause_does_not_reach_another():
    sub = {"expect": "value", "all_of": [["electronic"], ["manual", "paper"]], "sources": ["RFP"]}
    s, _ = score_field(sub, got("Electronic submission (PDF not required) or manual paper submission", "RFP.pdf"))
    assert s == 1.0
    bid = {"expect": "value", "all_of": [["Z-100"]], "sources": ["RFP"]}
    assert score_field(bid, got("Bid No Z-100", "RFP.pdf"))[0] == 1.0  # "No" = number, not negation


def test_null_expectation_still_accepts_stated_absence():
    exp = {"expect": "null", "null_values": ["not required", "none"]}
    assert score_field(exp, got("Not required"))[0] == 1.0
    assert score_field(exp, got("None ")) == (1.0, "correct: states absence / equivalent")
