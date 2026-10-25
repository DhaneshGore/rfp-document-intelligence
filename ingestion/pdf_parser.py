"""PDF parsing with Docling (layout + TableFormer), with a PyMuPDF fallback.

Output is a list of structural `Block`s with 1-based physical page numbers.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import get_settings
from ingestion.cleaner import join_wrapped_lines
from ingestion.ocr_fallback import FallbackOcr, page_profile, pages_needing_ocr
from jobcontrol import JobCancelled, check_cancelled, report_detail
from schemas import Block, BlockType

log = logging.getLogger(__name__)

_HEADING_LABELS = {"title", "section_header"}
_LIST_LABELS = {"list_item"}
# Page headers/footers are NOT skipped here: Docling sometimes labels real titles
# (e.g. "ADDENDUM No. 2") as page headers. Repeated furniture is removed later by
# cleaner.remove_repeated_furniture, which keeps the first occurrence.
_SKIP_LABELS = {"picture", "chart"}
_FURNITURE_LABELS = {"page_header", "page_footer"}


@dataclass
class PdfParseResult:
    blocks: list[Block]
    num_pages: int
    parser: str
    warnings: list[str] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)
    pdf_creation_date: Optional[str] = None
    pdf_title: Optional[str] = None
    ocr: bool = False
    ocr_fallback_pages: list[int] = field(default_factory=list)  # pages read by fallback OCR


def render_table_rows(grid: list[list[str]]) -> list[str]:
    """Render a cell grid as "a | b | c" rows.

    Merged cells are repeated by table extractors across every spanned column, so
    consecutive duplicate cells are collapsed. Empty rows are dropped.
    """
    rows: list[str] = []
    for row in grid:
        cells: list[str] = []
        for cell in row:
            c = " ".join((cell or "").split())
            if c and (not cells or cells[-1] != c):
                cells.append(c)
        if cells:
            line = " | ".join(cells)
            if not rows or rows[-1] != line:
                rows.append(line)
    return rows


def _table_block(rows: list[str], header_rows: int, page: Optional[int], caption: str = "") -> Optional[Block]:
    if not rows:
        return None
    header = "\n".join(rows[:header_rows]) if header_rows else None
    body = rows[header_rows:] if header_rows else rows
    if not body:  # header-only table -> treat whole thing as body
        header, body = None, rows
    text = "\n".join(([caption] if caption else []) + ([header] if header else []) + body)
    return Block(
        type=BlockType.TABLE, text=text, page=page, table_header=header, table_rows=body
    )


Box = tuple[float, float, float, float]  # left, top, right, bottom (top-left origin)


def _side_by_side(a: Box, b: Box) -> bool:
    """True if two boxes share a horizontal band but do not overlap horizontally."""
    v_overlap = min(a[3], b[3]) - max(a[1], b[1])
    h_overlap = min(a[2], b[2]) - max(a[0], b[0])
    min_h = max(min(a[3] - a[1], b[3] - b[1]), 1.0)
    return v_overlap > 0.5 * min_h and h_overlap <= 0


def reading_order(blocks: list[Block], boxes: list[Optional[Box]], furniture: list[bool]) -> list[Block]:
    """Re-sort single-column pages top-to-bottom.

    Docling sometimes emits a list group before interleaved paragraphs (e.g. numbered
    questions with indented answers), which swaps questions and answers. On pages
    where no two body blocks sit side by side, vertical position is the true reading
    order. Multi-column pages keep Docling's order. Blocks without a box keep their
    relative position.
    """
    by_page: dict[Optional[int], list[int]] = {}
    for i, b in enumerate(blocks):
        by_page.setdefault(b.page, []).append(i)
    order: list[int] = []
    for page in by_page:  # dict keeps first-seen page order
        idx = by_page[page]
        body = [boxes[i] for i in idx if boxes[i] is not None and not furniture[i]]
        multi_column = any(
            _side_by_side(a, b) for k, a in enumerate(body) for b in body[k + 1:]
        )
        if page is None or multi_column or any(boxes[i] is None for i in idx):
            order.extend(idx)
        else:
            order.extend(sorted(idx, key=lambda i: (round(boxes[i][1], 1), boxes[i][0])))
    return [blocks[i] for i in order]


_CODE_RE = re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9\-#./]{3,24}$")


def recover_code_tables(path: Path, min_rows: int = 5) -> dict[int, Block]:
    """Rebuild borderless "code column + description column" tables (SKU lists,
    part-number quotes) that layout models miss.

    A page qualifies when >= `min_rows` single-token code-like lines (e.g. 210-BLYZ)
    share one column. Every other line in the vertical span of that column is
    assigned to the code whose row band it starts in, giving "description | code"
    rows. Returns {page_number: table block}.
    """
    import fitz

    out: dict[int, Block] = {}
    with fitz.open(path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            lines = []
            for blk in page.get_text("dict")["blocks"]:
                for ln in blk.get("lines", []):
                    t = " ".join("".join(s["text"] for s in ln["spans"]).split())
                    if t:
                        x0, y0, x1, y1 = ln["bbox"]
                        lines.append((x0, y0, x1, y1, t))
            codes = [l for l in lines if _CODE_RE.match(l[4])]
            if len(codes) < min_rows:
                continue
            # the largest group of codes sharing a right edge (+-8pt) is the code column
            best: list = []
            for c in codes:
                group = [d for d in codes if abs(d[2] - c[2]) <= 8]
                if len(group) > len(best):
                    best = group
            if len(best) < min_rows:
                continue
            col = sorted(best, key=lambda l: l[1])
            col_l, col_r = min(c[0] for c in col), max(c[2] for c in col)
            top, bottom = col[0][1] - 4, col[-1][3] + 40
            starts = [c[1] - 4 for c in col]
            rows: list[list[str]] = [[] for _ in col]
            header_code, header_desc = "", []
            col_ids = {id(c) for c in col}
            for ln in sorted(lines, key=lambda l: (l[1], l[0])):
                if id(ln) in col_ids:
                    continue
                if ln[2] > col_l and ln[0] < col_r:  # same column but not a code: header
                    if ln[3] <= col[0][1] + 2:
                        header_code = ln[4]
                    continue
                if top - 25 <= ln[1] < top:  # just above the first row: column headers
                    header_desc.append(ln[4])
                    continue
                if not (top <= ln[1] <= bottom):
                    continue
                k = max(i for i, s in enumerate(starts) if s <= ln[1]) if ln[1] >= starts[0] else 0
                rows[k].append(ln[4])
            body = [f"{' '.join(r)} | {c[4]}" if r else c[4] for r, c in zip(rows, col)]
            header = None
            if header_code or header_desc:
                header = f"{' '.join(header_desc) or 'Description'} | {header_code or 'Code'}"
            text = "\n".join(([header] if header else []) + body)
            out[pno] = Block(type=BlockType.TABLE, text=text, page=pno, table_header=header, table_rows=body)
    return out


def merge_recovered_tables(blocks: list[Block], recovered: dict[int, Block]) -> list[Block]:
    """Replace the blocks a recovered table re-reads with the table itself.

    A page is skipped when an existing table already contains the codes. Other
    blocks on the page are dropped only if all their tokens appear in the table.
    """
    def toks(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    pending: dict[int, Block] = {}
    for page, table in recovered.items():
        codes = [r.rsplit(" | ", 1)[-1].lower() for r in table.table_rows or []]
        already = any(
            b.page == page and b.type == BlockType.TABLE
            and sum(c in b.text.lower() for c in codes) * 2 >= len(codes)
            for b in blocks
        )
        if not already:
            pending[page] = table
    if not pending:
        return blocks

    table_toks = {p: toks(t.text) for p, t in pending.items()}
    out: list[Block] = []
    inserted: set[int] = set()
    for b in blocks:
        p = b.page
        if p in pending and b.type != BlockType.TABLE:
            bt = toks(b.text)
            if bt and bt <= table_toks[p]:
                if p not in inserted:  # table goes where its first fragment was
                    out.append(pending[p])
                    inserted.add(p)
                continue
        out.append(b)
    for p, table in pending.items():  # nothing matched: append so content is not lost
        if p not in inserted:
            out.append(table)
    return out


class PdfParser:
    """Thread-safe, lazily-initialised Docling converter with a PyMuPDF fallback."""

    def __init__(
        self,
        ocr: Optional[bool] = None,
        table_structure: Optional[bool] = None,
        backend: Optional[str] = None,
    ):
        s = get_settings()
        self.backend = backend or s.pdf_parser_backend
        if self.backend not in ("docling", "pymupdf"):
            raise ValueError(f"Unknown PDF backend '{self.backend}' (use docling or pymupdf)")
        self.ocr = s.pdf_ocr_enabled if ocr is None else ocr
        self.table_structure = s.pdf_table_structure if table_structure is None else table_structure
        self.artifacts_dir = s.docling_artifacts_dir
        self.batch_pages = max(1, s.pdf_batch_pages)
        self._converters: dict[bool, object] = {}  # one converter per OCR setting
        self._lock = threading.Lock()
        self.fallback_ocr = FallbackOcr(Path(self.artifacts_dir) / "RapidOcr")

    # ------------------------------------------------------------------ docling
    def _get_converter(self, ocr: bool):
        if ocr not in self._converters:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            if not self.artifacts_dir.exists() or not any(self.artifacts_dir.iterdir()):
                # Download into a project-local folder (no symlinks -> works on Windows
                # without Developer Mode).
                from docling.utils.model_downloader import download_models

                log.info("Downloading Docling models to %s", self.artifacts_dir)
                download_models(
                    self.artifacts_dir,
                    with_code_formula=False,
                    with_picture_classifier=False,
                    with_rapidocr=True,
                )
            opts = PdfPipelineOptions(artifacts_path=str(self.artifacts_dir))
            opts.do_ocr = ocr
            opts.do_table_structure = self.table_structure
            self._converters[ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        return self._converters[ocr]

    def _parse_docling(self, path: Path, ocr: bool, total_pages: int = 0) -> tuple[list[Block], int]:
        """Convert with Docling in page batches.

        Batching (page numbers are preserved by Docling's `page_range`) lets a background
        job report page progress and stop at a batch boundary when cancelled.
        """
        from docling.datamodel.base_models import ConversionStatus

        if total_pages <= self.batch_pages:
            ranges: list[Optional[tuple[int, int]]] = [None]
        else:
            ranges = [(s, min(s + self.batch_pages - 1, total_pages))
                      for s in range(1, total_pages + 1, self.batch_pages)]
        blocks: list[Block] = []
        boxes: list[Optional[Box]] = []
        furniture: list[bool] = []
        num_pages = 0
        for rng in ranges:
            check_cancelled()
            kwargs = {"page_range": rng} if rng else {}
            with self._lock:  # Docling models are not guaranteed thread-safe
                res = self._get_converter(ocr).convert(str(path), raises_on_error=False, **kwargs)
            if res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                errs = "; ".join(str(e.error_message) for e in (res.errors or []))
                raise RuntimeError(f"Docling conversion failed ({res.status}): {errs}")
            b, bx, fu = self._extract(res.document)
            blocks += b
            boxes += bx
            furniture += fu
            num_pages = max(num_pages, rng[1] if rng else len(res.document.pages))
            if rng:
                report_detail(f"{path.name}: pages {rng[1]}/{total_pages}" + (" (OCR)" if ocr else ""))
        check_cancelled()
        return reading_order(blocks, boxes, furniture), max(num_pages, total_pages)

    @staticmethod
    def _extract(doc) -> tuple[list[Block], list[Optional[Box]], list[bool]]:
        """Docling document -> blocks (+ boxes for reading-order repair)."""
        from docling_core.types.doc import ContentLayer, TableItem

        blocks: list[Block] = []
        boxes: list[Optional[Box]] = []
        furniture: list[bool] = []

        def add(block: Block, item, label: str) -> None:
            blocks.append(block)
            furniture.append(label in _FURNITURE_LABELS)
            try:
                prov = item.prov[0]
                bb = prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height)
                boxes.append((bb.l, bb.t, bb.r, bb.b))
            except Exception:
                boxes.append(None)

        layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
        for item, level in doc.iterate_items(included_content_layers=layers):
            label = str(getattr(item, "label", "")).split(".")[-1].lower()
            if label in _SKIP_LABELS:
                continue
            page = item.prov[0].page_no if getattr(item, "prov", None) else None
            if isinstance(item, TableItem):
                grid = [[c.text for c in row] for row in item.data.grid]
                header_rows = 0
                for row in item.data.grid:
                    if row and all(getattr(c, "column_header", False) for c in row if c.text.strip()):
                        header_rows += 1
                    else:
                        break
                rows = render_table_rows(grid)
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:  # pragma: no cover - caption API differences
                    pass
                tb = _table_block(rows, min(header_rows, max(len(rows) - 1, 0)), page, caption)
                if tb:
                    add(tb, item, label)
                continue
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue
            if label in _FURNITURE_LABELS:
                btype = BlockType.TEXT
            elif label in _HEADING_LABELS and len(text) <= 200:
                btype = BlockType.HEADING
            elif label in _LIST_LABELS:
                btype = BlockType.LIST_ITEM
            else:
                btype = BlockType.TEXT
            add(Block(type=btype, text=text, page=page, level=max(int(level or 1), 1)), item, label)
        return blocks, boxes, furniture

    # ------------------------------------------------------------------ pymupdf
    @staticmethod
    def _parse_pymupdf(path: Path) -> tuple[list[Block], int]:
        import fitz  # PyMuPDF

        blocks: list[Block] = []
        with fitz.open(path) as pdf:
            for pno, page in enumerate(pdf, start=1):
                table_rects = []
                try:
                    for t in page.find_tables().tables:
                        rows = render_table_rows([[c or "" for c in r] for r in t.extract()])
                        tb = _table_block(rows, 1 if len(rows) > 1 else 0, pno)
                        if tb:
                            blocks.append(tb)
                            table_rects.append(fitz.Rect(t.bbox))
                except Exception as exc:  # table finder is best-effort
                    log.debug("PyMuPDF table detection failed on %s p%s: %s", path.name, pno, exc)
                for x0, y0, x1, y1, txt, *_ in page.get_text("blocks", sort=True):
                    r = fitz.Rect(x0, y0, x1, y1)
                    if any(r.intersects(tr) for tr in table_rects):
                        continue
                    txt = join_wrapped_lines(txt)
                    if txt.strip():
                        blocks.append(Block(type=BlockType.TEXT, text=txt, page=pno))
            return blocks, len(pdf)

    @staticmethod
    def _pdf_info(path: Path) -> tuple[Optional[str], Optional[str], dict[int, int]]:
        """Creation date, title and per-page character counts (for empty-page detection)."""
        try:
            import fitz

            with fitz.open(path) as pdf:
                md = pdf.metadata or {}
                chars = {i + 1: len(p.get_text().strip()) for i, p in enumerate(pdf)}
                return md.get("creationDate") or md.get("modDate"), md.get("title"), chars
        except Exception:
            return None, None, {}

    # ------------------------------------------------------------------ fallback OCR
    def _apply_ocr_fallback(self, path: Path, blocks: list[Block], warnings: list[str]) -> tuple[list[Block], list[int]]:
        """OCR only the scanned / mostly-image pages that came back (nearly) empty.

        A page's blocks are replaced only if the fallback read more text than the parser.
        Cancellation is checked before each page.
        """
        s = get_settings()
        if not s.ocr_fallback_enabled:
            return blocks, []
        try:
            profile = page_profile(path)
        except Exception as exc:  # unreadable for PyMuPDF too: nothing to render
            warnings.append(f"Fallback OCR skipped: {type(exc).__name__}: {exc}")
            return blocks, []
        pages = pages_needing_ocr(blocks, profile, s.ocr_fallback_min_chars, s.ocr_fallback_image_coverage)
        recovered: dict[int, list[Block]] = {}
        failed: list[int] = []
        error = ""
        for i, p in enumerate(pages, start=1):
            check_cancelled()
            report_detail(f"{path.name}: fallback OCR page {p} ({i}/{len(pages)})")
            try:
                got = self.fallback_ocr.ocr_page(path, p)
            except (ImportError, FileNotFoundError) as exc:  # engine unavailable: stop trying
                failed += pages[i - 1:]
                error = f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                log.warning("Fallback OCR failed on %s p%s: %s", path.name, p, exc)
                failed.append(p)
                error = error or f"{type(exc).__name__}: {exc}"
                continue
            before = sum(len(b.text.strip()) for b in blocks if b.page == p)
            if sum(len(b.text) for b in got) > before:
                recovered[p] = got
        check_cancelled()
        if recovered:
            warnings.append(f"Fallback OCR (RapidOCR on PyMuPDF page images) read page(s) {sorted(recovered)}")
            pending = sorted(recovered)
            out: list[Block] = []
            for b in blocks:  # keep document order: recovered pages go before the next page
                if b.page in recovered:
                    continue
                while pending and b.page is not None and b.page > pending[0]:
                    out += recovered[pending.pop(0)]
                out.append(b)
            for p in pending:
                out += recovered[p]
            blocks = out
        if failed:
            warnings.append(f"Fallback OCR failed on page(s) {failed[:20]}: {error}")
        return blocks, sorted(recovered)

    # ------------------------------------------------------------------ public
    def parse(self, path: Path, ocr: Optional[bool] = None) -> PdfParseResult:
        """Parse a PDF. `ocr` overrides the configured OCR setting for this call."""
        path = Path(path)
        use_ocr = self.ocr if ocr is None else ocr
        created, title, text_chars = self._pdf_info(path)
        warnings: list[str] = []
        parser = self.backend
        if self.backend == "pymupdf":
            blocks, num_pages = self._parse_pymupdf(path)
        else:
            try:
                blocks, num_pages = self._parse_docling(path, use_ocr, total_pages=len(text_chars))
                if not blocks and any(text_chars.values()):
                    raise RuntimeError("Docling returned no content although the PDF has a text layer")
            except JobCancelled:
                raise  # a user cancel must not turn into a PyMuPDF fallback
            except Exception as exc:
                log.warning("Docling failed for %s (%s); falling back to PyMuPDF", path.name, exc)
                warnings.append(f"Docling failed, used PyMuPDF fallback: {exc}")
                parser = "pymupdf"
                blocks, num_pages = self._parse_pymupdf(path)
        fallback_pages: list[int] = []
        if use_ocr:
            blocks, fallback_pages = self._apply_ocr_fallback(path, blocks, warnings)
        elif self.backend != "pymupdf":
            warnings.append("OCR was disabled for this document")
        return self._result(path, blocks, num_pages, parser, warnings, created, title, text_chars, use_ocr,
                            fallback_pages)

    def _result(self, path, blocks, num_pages, parser, warnings, created, title, text_chars,
                use_ocr: bool = False, fallback_pages: Optional[list[int]] = None) -> PdfParseResult:
        try:
            blocks = merge_recovered_tables(blocks, recover_code_tables(path))
        except Exception as exc:  # recovery is best-effort
            log.warning("Code-table recovery failed: %s", exc)
        pages_with_content = {b.page for b in blocks if b.page is not None}
        empty_pages = [p for p in range(1, num_pages + 1) if p not in pages_with_content]
        if empty_pages:
            no_text_layer = [p for p in empty_pages if text_chars.get(p, 0) < 20]
            msg = f"{len(empty_pages)} page(s) produced no text: {empty_pages[:20]}"
            if no_text_layer and not use_ocr:
                msg += " (no text layer - likely scanned/image pages; index with OCR to read them)"
            warnings.append(msg)
        return PdfParseResult(
            ocr=use_ocr,
            ocr_fallback_pages=fallback_pages or [],
            blocks=blocks,
            num_pages=num_pages,
            parser=parser,
            warnings=warnings,
            empty_pages=empty_pages,
            pdf_creation_date=created,
            pdf_title=title,
        )
