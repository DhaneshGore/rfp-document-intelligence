"""Text comparison helpers shared by extraction checks and the offline scorer (no other imports).

normalize_text   Unicode-safe form for substring comparison: NFKC (folds no-break, thin,
                 narrow and ideographic spaces, full-width digits, ligatures), every dash /
                 minus variant -> "-", zero-width characters removed, whitespace collapsed,
                 lower case.
mention_state    whether a text states something positively ("3-year warranty"), explicitly
                 says it is absent ("warranty not found in provided passages"), only hedges
                 ("warranty may apply"), or does not mention it at all.
"""
from __future__ import annotations

import re
import unicodedata

_DASHES_RE = re.compile(r"[‐-―−﹘﹣－]")
_ZERO_WIDTH_RE = re.compile(r"[​-‍⁠﻿­]")

POSITIVE, NEGATED, AMBIGUOUS, MISSING = "positive", "negated", "ambiguous", "missing"

# negation right before the term in the same clause: "no warranty", "not include a warranty"
_NEG_BEFORE_RE = re.compile(r"(?:(\w+)\s+)?\b(no|not|without|lacks?|lacking|none|n/a)\b(?![.#])[^.;:\n\d]{0,30}$",
                            re.I)
# "no" as an abbreviation of number: "Bid No JA-1", "Model No 5430"
_NUMBER_NOUNS = frozenset("bid model part item ref rfp rfq ifb contract lot line po account phone tel fax serial sku "
                          "order solicitation addendum page quote invoice".split())
# negation right after it: "warranty not found in provided passages", "warranty: none"
_NEG_AFTER_RE = re.compile(
    r"^[^.;\n\d]{0,40}?\b(not (found|stated|provided|specified|mentioned|listed|available|included|given|present|"
    r"identified|indicated|addressed)|none|n/a|unknown|absent|missing|unavailable|unspecified)\b", re.I)
# hedges that leave the item undecided (not "may"/"might": solicitations use them for permissions)
_HEDGE_RE = re.compile(r"\b(possibly|perhaps|unclear|uncertain|tbd|to be (determined|confirmed)|unconfirmed|"
                       r"not clear|appears? to)\b", re.I)
# a negation does not reach across these: parentheses, commas, "but"
_SCOPE_BREAK_RE = re.compile(r"[(),]|\bbut\b")
_CLAUSE_SPLIT_RE =re.compile(r"(?<=[.;!?])\s+|\n+|\s\|\s")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH_RE.sub("", _DASHES_RE.sub("-", text))
    return " ".join(text.lower().split())


def _negated_before(before: str) -> bool:
    before = _SCOPE_BREAK_RE.split(before)[-1]  # "(pdf not required) or manual" -> " or manual"
    m = _NEG_BEFORE_RE.search(before)
    return bool(m) and not (m.group(2) == "no" and (m.group(1) or "") in _NUMBER_NOUNS)


def mention_state(pattern: "re.Pattern | str", text: str) -> str:
    """positive | negated | ambiguous | missing for `pattern` (regex) in `text`.

    Sentence by sentence: an occurrence is negated when a negation sits next to it in the same
    clause, ambiguous when its clause only hedges, positive otherwise. One positive
    occurrence makes the text positive; else any hedge -> ambiguous; else negated / missing."""
    rx = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
    states = set()
    for sentence in _CLAUSE_SPLIT_RE.split(normalize_text(text)):
        for m in rx.finditer(sentence):
            before, after = sentence[:m.start()], sentence[m.end():]
            clause_after = _SCOPE_BREAK_RE.split(after)[0]
            clause = _SCOPE_BREAK_RE.split(before)[-1] + m.group(0) + clause_after
            if _negated_before(before) or _NEG_AFTER_RE.search(clause_after):
                states.add(NEGATED)
            elif _HEDGE_RE.search(clause):
                states.add(AMBIGUOUS)
            else:
                states.add(POSITIVE)
    for state in (POSITIVE, AMBIGUOUS, NEGATED):
        if state in states:
            return state
    return MISSING


def states_positively(pattern: "re.Pattern | str", text: str) -> bool:
    return mention_state(pattern, text) == POSITIVE
