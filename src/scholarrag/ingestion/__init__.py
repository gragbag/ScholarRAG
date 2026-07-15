"""Ingestion primitives — the pure transforms that turn files into chunks.

* :func:`content_hash` — idempotency key from raw bytes.
* :func:`extract_text` / :func:`detect_content_type` — bytes → plain text.
* :func:`chunk_text` (+ :class:`TextChunk`) — text → overlapping passages.

No DB, no embeddings, no async here — those are wired together by the ingestion
pipeline in the next step.
"""

from __future__ import annotations

from scholarrag.ingestion.chunk import TextChunk, chunk_text
from scholarrag.ingestion.hashing import content_hash
from scholarrag.ingestion.parse import (
    UnsupportedFileTypeError,
    detect_content_type,
    extract_text,
    extract_text_from_path,
)

# Imported last: pipeline depends on the primitives above.
from scholarrag.ingestion.pipeline import IngestionPipeline, IngestResult

__all__ = [
    "IngestResult",
    "IngestionPipeline",
    "TextChunk",
    "UnsupportedFileTypeError",
    "chunk_text",
    "content_hash",
    "detect_content_type",
    "extract_text",
    "extract_text_from_path",
]
