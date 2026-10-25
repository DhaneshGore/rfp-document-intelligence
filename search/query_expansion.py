"""Domain query expansion for common RFP vocabulary.

The expanded query is used for BM25 (where vocabulary mismatch hurts most); the
dense retriever sees the original question.
"""
from __future__ import annotations

import re

# trigger phrase -> extra terms appended to the keyword query
RFP_SYNONYMS: dict[str, list[str]] = {
    "deadline": ["due date", "submission deadline", "closing date", "solicitation due"],
    "due date": ["deadline", "closing date", "submission", "solicitation due"],
    "submission date": ["due date", "deadline", "closing date"],
    "closing date": ["due date", "deadline"],
    "submit": ["submission", "submittal", "electronic", "sealed"],
    "bid bond": ["bond", "bid security", "surety", "guarantee"],
    "bond": ["bid bond", "bid security", "surety"],
    "warranty": ["warranties", "guarantee", "extended warranty", "OEM"],
    "affidavit": ["affidavits", "affirm", "certification", "form"],
    "insurance": ["certificate of insurance", "liability", "coverage"],
    "contact": ["buyer", "procurement officer", "email", "phone", "POC"],
    "pre-bid": ["pre-proposal", "prebid conference", "pre-bid meeting"],
    "pre bid": ["pre-proposal", "prebid conference", "pre-bid meeting"],
    "meeting": ["conference", "pre-proposal"],
    "term": ["length of contract", "renewal", "initial term", "extension"],
    "contract length": ["term", "renewal", "initial term"],
    "payment": ["invoice", "invoicing", "net 30", "payment terms"],
    "delivery": ["deliver", "ship", "shipping", "lead time"],
    "specs": ["specifications", "minimum requirements", "configuration"],
    "specification": ["specifications", "minimum requirements", "requirements"],
    "price": ["pricing", "cost", "quote"],
    "evaluation": ["evaluation criteria", "points", "award"],
    "cooperative": ["interlocal", "purchasing cooperative", "piggyback", "master contract"],
    "addendum": ["addenda", "amendment", "changes"],
    "revised": ["addendum", "new", "extend", "extended", "changed"],
    "updated": ["addendum", "new", "changed"],
    "extended": ["addendum", "extend", "new"],
    "latest": ["addendum", "new"],
    "changed": ["addendum", "new", "revised"],
    "model": ["model number", "model #", "SKU"],
    "part number": ["SKU", "part no", "part #"],
}


_CHANGE_INTENT = re.compile(
    r"\b(revis\w*|updat\w*|latest|current|new|extend\w*|extension|chang\w*|amend\w*|"
    r"after (the |all )?(addend\w*|amendments?)|addend\w*)\b",
    re.I,
)


def has_change_intent(query: str) -> bool:
    """True if the question asks about a changed/latest value, i.e. where addendums matter."""
    return bool(_CHANGE_INTENT.search(query))


def expand_query(query: str) -> str:
    """Return the query with synonym terms appended (deduplicated)."""
    q = query.lower()
    extra: list[str] = []
    for trigger, terms in RFP_SYNONYMS.items():
        if re.search(rf"\b{re.escape(trigger)}\b", q):
            extra.extend(t for t in terms if t.lower() not in q and t not in extra)
    return f"{query} {' '.join(extra)}" if extra else query
