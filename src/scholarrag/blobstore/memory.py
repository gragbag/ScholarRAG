"""In-process blob store — tests and single-process (eager) use only.

Not shared across processes, so it can't back a separate Celery worker; that's
what :class:`LocalBlobStore` (filesystem) is for.
"""

from __future__ import annotations

from scholarrag.blobstore.base import BlobNotFoundError


class MemoryBlobStore:
    """A dict-backed :class:`BlobStore`."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data

    def get(self, key: str) -> bytes:
        try:
            return self._data[key]
        except KeyError as exc:
            raise BlobNotFoundError(key) from exc

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
