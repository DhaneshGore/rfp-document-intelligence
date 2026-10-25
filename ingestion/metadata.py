"""Document-level metadata extraction: doc type, addendum number, document date.

All rules are generic (filename + the opening text of the document) so that new,
unseen bid folders work without code changes.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from schemas import DocType

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ADDENDUM_RE = re.compile(
    r"\b(?:addendum|amendment)\s*(?:no\.?|number|num\.?|#)?\s*[-_:]?\s*(\d{1,3}|"
    + "|".join(_WORD_NUMBERS)
    + r")\b",
    re.I,
)
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_DATE_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ["%Y-%m-%d"]),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), ["%m/%d/%Y"]),
    (re.compile(rf"\b(\d{{1,2}})-({_MONTHS})[a-z]*-(\d{{4}})\b", re.I), ["%d-%b-%Y"]),
    (
        re.compile(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I),
        ["%B %d %Y", "%b %d %Y"],
    ),
]
_DATE_LABEL_RE = re.compile(
    r"(issue\s*date|date\s*issued|issued\s*on|publication(?:\s*date)?|posted|release\s*date"
    r"|addendum\s*date|dated|first\s*advertisement\s*date)",
    re.I,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_doc_id(bid_id: str, content_hash: str) -> str:
    """Deterministic document id: same bytes in the same bid -> same id (dedup)."""
    return hashlib.sha1(f"{bid_id}|{content_hash}".encode()).hexdigest()[:16]


def _head(text: str, n: int = 400) -> str:
    return re.sub(r"\s+", " ", text[:n]).lower()


def classify_doc_type(file_name: str, head_text: str = "") -> DocType:
    """Classify a document from its file name and the opening text.

    Order matters: an addendum file name usually also contains "RFP". HTML files get
    the addendum / affidavit / specs rules too (an addendum can be published as a web
    page), but otherwise stay `bid_page`: portal pages mention "bid" and "solicitation"
    everywhere, so the RFP rule would misfire on them.
    """
    name = Path(file_name).stem.lower().replace("_", " ").replace("-", " ")
    ext = Path(file_name).suffix.lower()
    head = _head(head_text)

    if re.search(r"\b(addendum|addenda|amendment)\b", name) or re.match(
        r"^(page \d+ \| \d+ )?(addendum|amendment)\b", head
    ):
        return DocType.ADDENDUM
    if "affidavit" in name or "affidavit" in head[:200]:
        return DocType.AFFIDAVIT
    if re.search(r"\b(spec|specs|specifications?|datasheet|data sheet|quote)\b", name) or re.search(
        r"\b(technical specifications?|product specifications?|spec sheet)\b", head[:200]
    ):
        return DocType.SPECS
    if ext in (".html", ".htm"):
        return DocType.BID_PAGE
    rfp_terms = (
        r"\b(rfp|rfq|rfi|ifb|itb|porfp|request for (proposals?|quotes?|quotations?|information)"
        r"|invitation (to|for) bids?|solicitation|tender|proposal|bid)\b"
    )
    if re.search(rfp_terms, name) or re.search(rfp_terms, head):
        return DocType.RFP
    return DocType.OTHER


def extract_addendum_number(file_name: str, head_text: str = "") -> Optional[int]:
    for source in (Path(file_name).stem.replace("_", " "), head_text[:600]):
        m = _ADDENDUM_RE.search(source)
        if m:
            tok = m.group(1).lower()
            return int(tok) if tok.isdigit() else _WORD_NUMBERS.get(tok)
    return None


def parse_date(text: str) -> Optional[str]:
    """Return the first date found in `text` as ISO yyyy-mm-dd."""
    best: Optional[tuple[int, str]] = None
    for pattern, fmts in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            cand = re.sub(r"\s+", " ", m.group(0).replace(",", "").replace(".", "")).title()
            cand = re.sub(r"\bSept\b", "Sep", cand)
            for fmt in fmts:
                try:
                    dt = datetime.strptime(cand, fmt)
                except ValueError:
                    continue
                if 1990 <= dt.year <= 2100 and (best is None or m.start() < best[0]):
                    best = (m.start(), dt.strftime("%Y-%m-%d"))
                break
    return best[1] if best else None


def extract_document_date(head_text: str) -> Optional[str]:
    """Find a labelled issue/publication date near the start of the document."""
    for m in _DATE_LABEL_RE.finditer(head_text):
        window = head_text[m.end() : m.end() + 60]
        d = parse_date(window)
        if d:
            return d
    return None


def pdf_metadata_date(raw: Optional[str]) -> Optional[str]:
    """Parse a PDF date string like "D:20240613175856-05'00'"."""
    if not raw:
        return None
    m = re.search(r"(\d{4})(\d{2})(\d{2})", raw)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
    except ValueError:
        return None
