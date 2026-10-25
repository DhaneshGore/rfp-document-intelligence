"""Text cleaning: unicode/whitespace normalisation, hyphenation, form-filler lines and
repeated running headers/footers."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from schemas import Block, BlockType

_FILLER_RUN = re.compile(r"[\.…_\-=–—~*]{5,}")  # "......", "_____", "-----"
_SPACES = re.compile(r"[ \t  -​  　]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n\s*(\w)")
_MULTI_NL = re.compile(r"\n{3,}")
_PAGE_MARKER = re.compile(
    r"^\s*(page\s*\d+\s*(of|\|)\s*\d+|page\s*\|?\s*\d+|\d+\s*of\s*\d+|-\s*\d+\s*-)\s*$", re.I
)


def normalize_text(text: str) -> str:
    """Normalise unicode, whitespace, hyphenated line breaks and filler runs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep the hyphen (compound words like "pre-proposal" are common in RFPs) but
    # re-join the word that was split across lines.
    text = _HYPHEN_BREAK.sub(r"\1-\2", text)
    text = _FILLER_RUN.sub(" ", text)
    text = _SPACES.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def join_wrapped_lines(text: str) -> str:
    """Join lines that were hard-wrapped by the PDF layout into paragraphs.

    Blank lines and bullet/numbered lines start a new paragraph.
    """
    out: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        starts_item = bool(re.match(r"^([•●▪\-*]|\d+[\.\)]|[a-zA-Z][\.\)])\s", s))
        if out and out[-1] != "" and not starts_item and not out[-1].endswith((":",)):
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return "\n".join(p for p in out).strip()


def is_page_marker(text: str) -> bool:
    return bool(_PAGE_MARKER.match(text.strip()))


def _signature(text: str) -> str:
    """Digit-insensitive signature so "Page 3 of 40" and "Page 4 of 40" match."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def _is_furniture(by_page: dict[int, str]) -> bool:
    """Identical text on many pages, or text whose only difference is a number that
    increases page by page (a page counter such as "Page 3 of 40")."""
    variants = set(by_page.values())
    if len(variants) == 1:
        return True
    nums = []
    for page in sorted(by_page):
        found = re.findall(r"\d+", by_page[page])
        if not found:
            return False
        nums.append(tuple(int(n) for n in found))
    # find the position that varies and require it to be strictly increasing
    for idx in range(len(nums[0])):
        col = [n[idx] for n in nums if len(n) > idx]
        if len(col) == len(nums) and len(set(col)) > 1:
            return all(a < b for a, b in zip(col, col[1:]))
    return False


def remove_repeated_furniture(
    blocks: list[Block], num_pages: int | None, min_pages: int = 3, ratio: float = 0.3
) -> tuple[list[Block], list[str]]:
    """Drop short blocks that repeat on many pages (running headers/footers).

    Returns the kept blocks and the removed signatures (for logging/tests).
    """
    occurrences: dict[str, dict[int, str]] = defaultdict(dict)  # sig -> page -> first text
    for b in blocks:
        if b.page is not None and b.type != BlockType.TABLE and len(b.text) <= 150:
            occurrences[_signature(b.text)].setdefault(b.page, b.text)
    distinct_pages = len({b.page for b in blocks if b.page is not None}) or (num_pages or 0)
    threshold = max(min_pages, int(ratio * distinct_pages + 0.999))
    repeated = {
        sig for sig, by_page in occurrences.items()
        if len(by_page) >= threshold and _is_furniture(by_page)
    }

    # The first occurrence is kept: on page 1 a repeated line is often the real title
    # (e.g. "ADDENDUM No. 1"); only the repeats on later pages are furniture.
    kept: list[Block] = []
    seen: set[str] = set()
    for b in blocks:
        if b.type != BlockType.TABLE:
            if is_page_marker(b.text):
                continue
            sig = _signature(b.text)
            if b.page is not None and sig in repeated:
                if sig in seen:
                    continue
                seen.add(sig)
        kept.append(b)
    return kept, sorted(repeated)


def clean_blocks(blocks: list[Block], num_pages: int | None = None) -> list[Block]:
    """Normalise every block, drop empties and running headers/footers."""
    cleaned: list[Block] = []
    for b in blocks:
        text = normalize_text(b.text)
        if not text:
            continue
        update: dict = {"text": text}
        if b.type == BlockType.TABLE:
            update["table_header"] = normalize_text(b.table_header or "") or None
            update["table_rows"] = [r for r in (normalize_text(x) for x in b.table_rows or []) if r]
        cleaned.append(b.model_copy(update=update))
    cleaned, _ = remove_repeated_furniture(cleaned, num_pages)
    return cleaned
