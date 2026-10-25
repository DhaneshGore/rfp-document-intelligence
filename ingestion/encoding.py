"""Decode saved HTML pages to text.

Order (close to what browsers do for a local file): byte-order mark, UTF-16 without a
BOM (NUL-byte pattern), the charset declared in <meta>, strict UTF-8, then Windows-1252
(the de-facto default for older pages and Word "Save as Web Page" files).
"""
from __future__ import annotations

import codecs
import re

_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:\-]+)""", re.I)
# Per the HTML spec these labels mean windows-1252.
_AS_CP1252 = {"iso-8859-1", "iso8859-1", "latin-1", "latin1", "l1", "us-ascii", "ascii", "cp819", "windows-1252"}
_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def detect_html_encoding(raw: bytes) -> str:
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return enc
    head = raw[:2048]
    if len(head) >= 4 and head.count(b"\x00") >= len(head) // 3:  # UTF-16 without BOM
        return "utf-16-le" if head[1::2].count(b"\x00") > head[0::2].count(b"\x00") else "utf-16-be"
    m = _META_CHARSET.search(raw[:8192])
    if m:
        label = m.group(1).decode("ascii", "ignore").lower()
        label = "cp1252" if label in _AS_CP1252 else label
        try:
            enc = codecs.lookup(label).name
            if enc not in ("utf-16", "utf-16-le", "utf-16-be", "utf-32"):  # a NUL-free file can't be UTF-16
                raw.decode(enc)
                return enc
        except (LookupError, UnicodeDecodeError):
            pass  # unknown label, or declared charset is wrong: fall through
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def decode_html(raw: bytes) -> str:
    return raw.decode(detect_html_encoding(raw), errors="replace")
