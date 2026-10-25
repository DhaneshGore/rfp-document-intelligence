"""Heading-aware, table-aware chunking for RFP documents.

* Text is grouped under the current section heading; a new heading closes the chunk
  so unrelated sections are never mixed.
* Inside a section, blocks are packed up to `chunk_size` characters; when a chunk
  overflows, the tail of the previous chunk (`chunk_overlap` chars, cut on a sentence
  or word boundary) is carried into the next one.
* Tables become their own chunks. Tables up to `max_table_chunk_size` stay whole;
  larger tables are split on row boundaries with the header row repeated.
* Each chunk's indexed text is prefixed with its section heading for context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config import get_settings
from ingestion.metadata import make_doc_id
from schemas import Block, BlockType, Chunk, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\"'])")
# Table captions: "Table 5: ...", "Table A-2. ...", "Exhibit 3 - ...", or a bare short "Table 1".
# Figure captions are not used: figures are dropped, so their caption may sit right
# before an unrelated table.
_CAPTION_RE = re.compile(
    r"^(table|tbl\.?|exhibit|schedule)\s+([A-Z]?\d+(?:[.\-]\d+)*[a-z]?)\b\s*([:.\-–—)])?", re.I
)
_CAPTION_START = re.compile(r"\b(?:table|exhibit|schedule)\s+[A-Z]?\d+(?:[.\-]\d+)*[a-z]?\s*[:.\-–—]", re.I)
_MAX_CAPTION = 300


@dataclass
class _Piece:
    text: str
    page: Optional[int]


def _split_long(text: str, size: int) -> list[str]:
    """Split an over-long paragraph into sentence-packed pieces of <= size chars."""
    if len(text) <= size:
        return [text]
    parts, cur = [], ""
    for sent in _SENTENCE_SPLIT.split(text):
        while len(sent) > size:  # a single giant "sentence": hard-wrap on words
            cut = sent.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > size:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def _tail(text: str, n: int) -> str:
    """Last ~n chars of text, starting on a sentence (or at least word) boundary."""
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    window = text[-n:]
    m = re.search(r"[.!?;:]\s+", window)
    if m and m.end() < len(window) - 20:
        return window[m.end():].strip()
    sp = window.find(" ")
    return window[sp + 1:].strip() if sp != -1 else window


def _take_caption(buf: list[_Piece], table: Block) -> Optional[str]:
    """Pop a caption ("Table 5: ...") that directly precedes `table` on the same page.

    Layout parsers often emit a table's caption as a separate paragraph, which would end
    up in a different chunk than the table rows, so neither chunk says which table the
    values belong to. The caption is attached only when it is the last text before the
    table, is on the same page, and the table does not already carry that label. If a
    second caption was merged into the same paragraph ("Table 6: ... Table 7: ..."), only
    the first is attached; the rest stays as text.
    """
    if not buf or buf[-1].page != table.page:
        return None
    text = buf[-1].text.strip()
    m = _CAPTION_RE.match(text)
    if not m or len(text) > _MAX_CAPTION or (not m.group(3) and (len(text) > 100 or text.endswith("."))):
        return None
    if re.search(rf"\b{re.escape(m.group(1))}\s*{re.escape(m.group(2))}\b", table.text, re.I):
        return None  # the parser already attached this caption to the table
    nxt = _CAPTION_START.search(text, m.end())
    if nxt:
        caption, rest = text[: nxt.start()].strip(), text[nxt.start():].strip()
        buf[-1] = _Piece(rest, buf[-1].page)
    else:
        caption = text
        buf.pop()
    return caption


class Chunker:
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_table_chunk_size: Optional[int] = None,
    ):
        s = get_settings()
        self.chunk_size = chunk_size or s.chunk_size
        self.chunk_overlap = s.chunk_overlap if chunk_overlap is None else chunk_overlap
        self.max_table = max_table_chunk_size or s.max_table_chunk_size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        md = doc.metadata
        doc_id = make_doc_id(md.bid_id, md.content_hash)
        chunks: list[Chunk] = []
        headings: dict[int, str] = {}
        buf: list[_Piece] = []

        def section() -> Optional[str]:
            if not headings:
                return None
            ordered = [headings[k] for k in sorted(headings)]
            return " > ".join(ordered[-2:])

        def emit(text: str, pages: list[Optional[int]], ctype: str) -> None:
            body = text.strip()
            if len(body) < 3:
                return
            sec = section()
            indexed = f"{sec}\n{body}" if sec and not body.startswith(sec) else body
            known = [p for p in pages if p is not None]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{len(chunks):04d}",
                    doc_id=doc_id,
                    text=indexed,
                    bid_id=md.bid_id,
                    file_name=md.file_name,
                    source_path=md.source_path,
                    doc_type=md.doc_type,
                    addendum_number=md.addendum_number,
                    page_number=min(known) if known else None,
                    page_end=max(known) if known else None,
                    document_date=md.document_date,
                    section=sec,
                    chunk_type=ctype,
                    chunk_index=len(chunks),
                )
            )

        def flush(carry_overlap: bool) -> None:
            nonlocal buf
            if not buf:
                return
            text = "\n".join(p.text for p in buf)
            emit(text, [p.page for p in buf], "text")
            # Short chunks are not carried over: they would just be duplicated whole.
            if carry_overlap and self.chunk_overlap and len(text) > 2 * self.chunk_overlap:
                tail = _tail(text, self.chunk_overlap)
                buf = [_Piece(tail, buf[-1].page)] if tail else []
            else:
                buf = []

        def buf_len() -> int:
            return sum(len(p.text) + 1 for p in buf)

        for block in doc.blocks:
            if block.type == BlockType.HEADING:
                level = block.level
                if headings.get(level) == block.text:
                    continue  # running header repeated on every page
                flush(carry_overlap=False)
                headings = {k: v for k, v in headings.items() if k < level}
                headings[level] = block.text
                continue

            if block.type == BlockType.TABLE:
                caption = _take_caption(buf, block)
                flush(carry_overlap=False)
                self._emit_table(block, emit, caption)
                continue

            text = f"- {block.text}" if block.type == BlockType.LIST_ITEM else block.text
            for piece in _split_long(text, self.chunk_size):
                if buf and buf_len() + len(piece) > self.chunk_size:
                    flush(carry_overlap=True)
                buf.append(_Piece(piece, block.page))
        flush(carry_overlap=False)
        return chunks

    def _emit_table(self, block: Block, emit, caption: Optional[str] = None) -> None:
        text = f"{caption}\n{block.text}" if caption else block.text
        if len(text) <= self.max_table or not block.table_rows:
            emit(text, [block.page], "table")
            return
        # split tables repeat the caption and the header row in every part
        header = "\n".join(x for x in (caption, block.table_header) if x)
        group: list[str] = []
        size = len(header)
        for row in block.table_rows:
            row_parts = _split_long(row, self.max_table - len(header) - 1)
            for r in row_parts:
                if group and size + len(r) + 1 > self.max_table:
                    emit("\n".join(([header] if header else []) + group), [block.page], "table")
                    group, size = [], len(header)
                group.append(r)
                size += len(r) + 1
        if group:
            emit("\n".join(([header] if header else []) + group), [block.page], "table")
