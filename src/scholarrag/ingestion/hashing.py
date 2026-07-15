"""Content hashing — the idempotency key.

We hash the *raw file bytes* (not the extracted text) so that re-uploading the
exact same file is detected before any expensive parsing/embedding happens. The
hash maps to ``documents.content_hash`` (a unique column), so the database
enforces "ingest each distinct file at most once".
"""

from __future__ import annotations

import hashlib


def content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data`` (64 hex chars)."""
    return hashlib.sha256(data).hexdigest()
