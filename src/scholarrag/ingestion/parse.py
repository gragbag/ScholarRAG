"""Text extraction — raw file bytes → plain text.

Dispatches on content type:

* ``txt`` / ``md`` — decoded as UTF-8 (markdown is human-readable; we keep it
  as-is for now — no markdown library needed).
* ``pdf``          — text extracted per page with pypdf.

Operates on ``bytes`` so it matches the upload endpoint's input; a small
``extract_text_from_path`` helper wraps it for the seed script.

Limitation (documented, not a bug): scanned/image-only PDFs have no text layer,
so pypdf returns little or nothing — OCR is out of scope for Phase 1.
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

ContentType = str  # "pdf" | "md" | "txt"

# Map file extensions to a normalised content type.
_EXT_TO_TYPE: dict[str, ContentType] = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".text": "txt",
}


class UnsupportedFileTypeError(ValueError):
    """Raised for a file extension / content type we don't know how to parse."""


def detect_content_type(filename: str) -> ContentType:
    """Infer the content type from a filename's extension."""
    ext = Path(filename).suffix.lower()
    try:
        return _EXT_TO_TYPE[ext]
    except KeyError:
        raise UnsupportedFileTypeError(
            f"unsupported file extension: {ext!r} ({filename})"
        ) from None


def extract_text(data: bytes, content_type: ContentType) -> str:
    """Extract plain text from ``data`` given its ``content_type``."""
    if content_type in ("txt", "md"):
        return data.decode("utf-8", errors="replace")
    if content_type == "pdf":
        return _extract_pdf(data)
    raise UnsupportedFileTypeError(f"unsupported content type: {content_type!r}")


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_text_from_path(path: str | Path) -> str:
    """Convenience: read a file and extract its text (used by the seed script)."""
    p = Path(path)
    content_type = detect_content_type(p.name)
    return extract_text(p.read_bytes(), content_type)
