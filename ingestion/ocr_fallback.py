"""Fallback OCR for pages the main PDF parser left (nearly) empty.

Docling's own OCR normally reads scanned pages. When it does not - the conversion
failed and the PyMuPDF fallback (no OCR) was used, or Docling returned almost nothing
for a scanned / mostly-image page - those pages only are rendered with PyMuPDF and
read with RapidOCR directly, using the same local PP-OCR models Docling downloaded.
"""
from __future__ import annotations

import logging
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import get_settings
from schemas import Block, BlockType

log = logging.getLogger(__name__)

# Model files Docling's downloader puts in <docling_artifacts_dir>/RapidOcr
_MODEL_FILES = {
    "Det.model_path": "PP-OCRv6_det_small.onnx",
    "Cls.model_path": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "Rec.model_path": "PP-OCRv6_rec_small.onnx",
    "Rec.rec_keys_path": "ppocrv6_dict.txt",
}

Segment = tuple[float, float, float, float, str]  # left, top, right, bottom, text


@dataclass
class PageInfo:
    text_chars: int  # characters in the PDF text layer
    image_coverage: float  # share of the page area covered by images (0..1)


def page_profile(path: Path) -> dict[int, PageInfo]:
    import fitz

    out: dict[int, PageInfo] = {}
    with fitz.open(path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            area = abs(page.rect) or 1.0
            covered = 0.0
            for info in page.get_image_info():
                covered += abs(fitz.Rect(info["bbox"]) & page.rect)
            out[pno] = PageInfo(len(page.get_text().strip()), min(covered / area, 1.0))
    return out


def pages_needing_ocr(
    blocks: list[Block], profile: dict[int, PageInfo], min_chars: int, min_coverage: float
) -> list[int]:
    """Pages that are scanned (no text layer) or mostly image, but produced < min_chars."""
    got: dict[int, int] = {}
    for b in blocks:
        if b.page is not None:
            got[b.page] = got.get(b.page, 0) + len(b.text.strip())
    return [
        p for p, info in sorted(profile.items())
        if got.get(p, 0) < min_chars and (info.text_chars < 20 or info.image_coverage >= min_coverage)
    ]


def segments_to_blocks(segments: list[Segment], page: int) -> list[Block]:
    """Group recognised segments into lines (shared vertical band), lines into paragraphs.

    Segments on one line separated by a wide gap (form label / value, table cells) are
    joined with " | " so they stay associated; a vertical gap larger than ~0.8 line
    height starts a new paragraph block.
    """
    if not segments:
        return []
    lines: list[list] = []  # [top, bottom, [segments]]
    for s in sorted(segments, key=lambda s: ((s[1] + s[3]) / 2, s[0])):
        cy = (s[1] + s[3]) / 2
        if lines and lines[-1][0] <= cy <= lines[-1][1]:
            lines[-1][2].append(s)
            lines[-1][0], lines[-1][1] = min(lines[-1][0], s[1]), max(lines[-1][1], s[3])
        else:
            lines.append([s[1], s[3], [s]])
    height = statistics.median(b - t for t, b, _ in lines) or 1.0

    def line_text(segs: list[Segment]) -> str:
        parts: list[str] = []
        prev_right: Optional[float] = None
        for left, _, right, _, text in sorted(segs, key=lambda s: s[0]):
            if prev_right is not None:
                parts.append(" | " if left - prev_right > 2 * height else " ")
            parts.append(text)
            prev_right = right
        return "".join(parts)

    blocks: list[Block] = []
    para: list[str] = []
    prev_bottom: Optional[float] = None
    for top, bottom, segs in lines:
        if para and prev_bottom is not None and top - prev_bottom > 0.8 * height:
            blocks.append(Block(type=BlockType.TEXT, text="\n".join(para), page=page))
            para = []
        para.append(line_text(segs))
        prev_bottom = bottom
    if para:
        blocks.append(Block(type=BlockType.TEXT, text="\n".join(para), page=page))
    return blocks


class FallbackOcr:
    """Lazily-initialised RapidOCR engine reading PyMuPDF page renders."""

    def __init__(self, models_dir: Optional[Path] = None):
        s = get_settings()
        self.models_dir = Path(models_dir or Path(s.docling_artifacts_dir) / "RapidOcr")
        self.dpi = s.ocr_fallback_dpi
        self.min_score = s.ocr_fallback_min_score
        self._engine = None
        self._lock = threading.Lock()

    def _get_engine(self):
        if self._engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import EngineType

            missing = [n for n in _MODEL_FILES.values() if not (self.models_dir / n).is_file()]
            if missing:
                raise FileNotFoundError(f"RapidOCR models missing in {self.models_dir}: {missing}")
            params: dict[str, object] = {"Global.log_level": "warning"}
            params.update({k: str(self.models_dir / v) for k, v in _MODEL_FILES.items()})
            for task in ("Det", "Cls", "Rec"):
                params[f"{task}.engine_type"] = EngineType.ONNXRUNTIME
            self._engine = RapidOCR(params=params)
        return self._engine

    def recognise(self, image) -> list[Segment]:
        """RGB numpy image -> segments with confidence >= min_score."""
        with self._lock:  # one ONNX session set, not guaranteed thread-safe
            res = self._get_engine()(image)
        boxes = res.boxes if res.boxes is not None else []
        out: list[Segment] = []
        for box, text, score in zip(boxes, res.txts or (), res.scores or ()):
            text = " ".join(str(text).split())
            if text and score >= self.min_score:
                xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
                out.append((min(xs), min(ys), max(xs), max(ys), text))
        return out

    def ocr_page(self, path: Path, page_no: int) -> list[Block]:
        import fitz
        import numpy as np

        with fitz.open(path) as pdf:
            pix = pdf[page_no - 1].get_pixmap(dpi=self.dpi, alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            img = np.repeat(img, 3, axis=2)
        return segments_to_blocks(self.recognise(np.ascontiguousarray(img[:, :, :3])), page_no)
