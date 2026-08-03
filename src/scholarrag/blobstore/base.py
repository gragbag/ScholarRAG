"""BlobStore protocol — raw uploaded bytes live in object storage, not the DB.

Big blobs (PDFs) don't belong in Postgres: they'd fill the free-tier database and
bloat backups. So the raw bytes go to a blob store and the ``documents`` row keeps
only a short ``blob_key``. Like the vector store / embedder, it's a swappable
interface:

* :class:`MemoryBlobStore` — in-process dict (tests, single-process).
* :class:`LocalBlobStore`  — local filesystem (dev; shared across API + worker).
* :class:`S3BlobStore`     — S3-compatible object storage: R2 / AWS S3 / MinIO.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BlobNotFoundError(KeyError):
    """The requested blob key does not exist in the store."""


@runtime_checkable
class BlobStore(Protocol):
    """Store and retrieve raw bytes by a string key. Construction does no I/O."""

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` (overwrites if it exists)."""
        ...

    def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key``; raise :class:`BlobNotFoundError` if absent."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key`` (a no-op if it doesn't exist)."""
        ...
