"""Filesystem blob store — local dev.

Writes each blob to ``<root>/<key>``. Because it's on disk, it's **shared across
processes on the same host**, so your API can ``put`` a blob and a separate
``make worker`` can ``get`` it (an in-memory store couldn't). Not for Cloud Run,
where the filesystem is ephemeral and per-instance — use S3/R2 there.
"""

from __future__ import annotations

from pathlib import Path

from scholarrag.blobstore.base import BlobNotFoundError


class LocalBlobStore:
    """A directory-backed :class:`BlobStore`."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise BlobNotFoundError(key) from exc

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
