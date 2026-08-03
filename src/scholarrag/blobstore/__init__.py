"""Blob storage — raw uploaded bytes, kept out of Postgres.

Use :func:`build_blob_store` to get the backend implied by ``BLOB_BACKEND``:
``memory`` (tests), ``local`` filesystem (dev), or ``s3`` (R2 / AWS S3 / MinIO).
"""

from __future__ import annotations

from scholarrag.blobstore.base import BlobNotFoundError, BlobStore
from scholarrag.blobstore.local import LocalBlobStore
from scholarrag.blobstore.memory import MemoryBlobStore
from scholarrag.config import Settings, get_settings

__all__ = [
    "BlobNotFoundError",
    "BlobStore",
    "LocalBlobStore",
    "MemoryBlobStore",
    "build_blob_store",
]


def build_blob_store(settings: Settings | None = None) -> BlobStore:
    """Return the blob store implied by configuration (``BLOB_BACKEND``)."""
    settings = settings or get_settings()
    if settings.blob_backend == "memory":
        return MemoryBlobStore()
    if settings.blob_backend == "s3":
        from scholarrag.blobstore.s3 import S3BlobStore

        return S3BlobStore(settings)
    return LocalBlobStore(settings.blob_local_dir)
