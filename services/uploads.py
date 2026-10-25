"""Validation and safe storage of user-uploaded files."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from config import get_settings
from ingestion.encoding import decode_html

_BID_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
_HTML_SNIFF = re.compile(r"<\s*(!doctype\s+html|html|head|body|div|p|table|title)\b", re.I)


class UploadError(ValueError):
    pass


def validate_bid_id(bid_id: str) -> str:
    bid_id = (bid_id or "").strip()
    if not _BID_ID_RE.match(bid_id):
        raise UploadError(
            "Invalid bid id: use 1-64 letters, digits, '-' or '_' (must start with a letter or digit)"
        )
    return bid_id


def safe_filename(name: str, allowed_ext: list[str] | None = None) -> str:
    """Strip any directory part and unsafe characters; enforce an allowed extension."""
    allowed_ext = allowed_ext or get_settings().allowed_extensions
    name = unicodedata.normalize("NFKC", name or "")
    name = re.split(r"[\\/]", name)[-1]  # drop directories (both separators)
    name = re.sub(r"[^A-Za-z0-9._ ()\-{}#&,+]", "_", name)
    name = re.sub(r"_+", "_", name).strip(" ._")
    stem, dot, ext = name.rpartition(".")
    if not dot or not stem:
        raise UploadError("File name must have a name and an extension")
    ext = f".{ext.lower()}"
    if ext not in allowed_ext:
        raise UploadError(f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_ext)}")
    if stem.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?i)(com|lpt)\d", stem):
        stem = f"_{stem}"
    return f"{stem[:150]}{ext}"


def validate_content(file_name: str, data: bytes, max_mb: int | None = None) -> None:
    max_mb = max_mb or get_settings().max_upload_mb
    if not data:
        raise UploadError(f"'{file_name}' is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"'{file_name}' exceeds the {max_mb} MB upload limit")
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        if b"%PDF-" not in data[:1024]:
            raise UploadError(f"'{file_name}' is not a valid PDF (missing %PDF header)")
    elif ext in (".html", ".htm"):
        text = decode_html(data[:65536])  # handles UTF-16 / UTF-8 BOMs and declared charsets
        if "\x00" in text[:4096] or not _HTML_SNIFF.search(text):
            raise UploadError(f"'{file_name}' does not look like an HTML document")


def inspect_pdf(path: Path) -> dict:
    """Cheap pre-indexing profile of a PDF (PyMuPDF, no layout models).

    Used to warn before indexing very large documents: pages, embedded images, pages
    without a text layer (need OCR), a rough indexing-time estimate with OCR, and
    whether the file exceeds LARGE_PDF_PAGES / LARGE_PDF_IMAGES.
    """
    s = get_settings()
    import fitz

    with fitz.open(path) as pdf:
        pages = len(pdf)
        images = sum(len(p.get_images(full=False)) for p in pdf)
        scanned = sum(1 for p in pdf if len(p.get_text().strip()) < 20)
    return {
        "pages": pages,
        "images": images,
        "pages_without_text": scanned,
        "estimated_seconds_with_ocr": round(pages * s.ocr_seconds_per_page),
        "large": pages > s.large_pdf_pages or images > s.large_pdf_images,
    }


def store_upload(bid_id: str, original_name: str, data: bytes) -> tuple[Path, bool]:
    """Validate and write an upload to uploads/<bid_id>/<safe name>.

    Returns (path, already_existed). Identical content already stored under the same
    name is not rewritten; a different file with the same name gets a hash suffix.
    """
    s = get_settings()
    bid_id = validate_bid_id(bid_id)
    name = safe_filename(original_name)
    validate_content(name, data)

    root = s.uploads_dir.resolve()
    target_dir = (root / bid_id).resolve()
    if root not in target_dir.parents:
        raise UploadError("Invalid upload location")
    target_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(data).hexdigest()
    target = (target_dir / name).resolve()
    if target.parent != target_dir:
        raise UploadError("Invalid file name")
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            return target, True
        target = target_dir / f"{target.stem}_{digest[:8]}{target.suffix}"
        if target.exists():
            return target, True
    target.write_bytes(data)
    return target, False
