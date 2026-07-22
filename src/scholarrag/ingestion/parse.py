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
import re
from pathlib import Path

from pypdf import PdfReader

ContentType = str  # "pdf" | "md" | "txt"

# Control characters to strip from extracted text: NUL (which Postgres text
# columns reject outright) plus the other C0 controls PDF glyph extraction leaves
# behind — but keep tab (09), newline (0a), carriage-return (0d).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

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
    """Extract plain text from ``data`` given its ``content_type`` (sanitized)."""
    if content_type in ("txt", "md"):
        text = data.decode("utf-8", errors="replace")
    elif content_type == "pdf":
        text = _extract_pdf(data)
    else:
        raise UnsupportedFileTypeError(f"unsupported content type: {content_type!r}")
    # PDF glyph extraction can emit NUL/control bytes; strip them so the text is
    # storable in Postgres and clean for embedding.
    return _CONTROL_CHARS_RE.sub("", text)


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_text_from_path(path: str | Path) -> str:
    """Convenience: read a file and extract its text (used by the seed script)."""
    p = Path(path)
    content_type = detect_content_type(p.name)
    return extract_text(p.read_bytes(), content_type)
