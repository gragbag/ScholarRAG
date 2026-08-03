"""Blob-store tests — memory/local round-trips, factory selection, S3 exercise."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from scholarrag.blobstore import (
    BlobNotFoundError,
    BlobStore,
    LocalBlobStore,
    MemoryBlobStore,
    build_blob_store,
)
from scholarrag.blobstore.s3 import S3BlobStore
from scholarrag.config import Settings


def _roundtrip(store: BlobStore) -> None:
    store.put("documents/abc", b"hello pdf bytes")
    assert store.get("documents/abc") == b"hello pdf bytes"
    store.put("documents/abc", b"overwritten")  # overwrite in place
    assert store.get("documents/abc") == b"overwritten"
    store.delete("documents/abc")
    with pytest.raises(BlobNotFoundError):
        store.get("documents/abc")
    store.delete("documents/abc")  # deleting a missing key is a no-op


def test_memory_blob_store_roundtrip() -> None:
    _roundtrip(MemoryBlobStore())


def test_local_blob_store_roundtrip(tmp_path: Path) -> None:
    _roundtrip(LocalBlobStore(tmp_path))


def test_stores_satisfy_protocol(tmp_path: Path) -> None:
    assert isinstance(MemoryBlobStore(), BlobStore)
    assert isinstance(LocalBlobStore(tmp_path), BlobStore)
    assert isinstance(S3BlobStore(Settings(_env_file=None)), BlobStore)


def test_build_blob_store_selects_backend(tmp_path: Path) -> None:
    memory = build_blob_store(Settings(_env_file=None, blob_backend="memory"))
    assert isinstance(memory, MemoryBlobStore)
    local = build_blob_store(
        Settings(_env_file=None, blob_backend="local", blob_local_dir=str(tmp_path))
    )
    assert isinstance(local, LocalBlobStore)
    assert isinstance(build_blob_store(Settings(_env_file=None, blob_backend="s3")), S3BlobStore)


class _FakeS3Client:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.store.pop((Bucket, Key), None)


def test_s3_blob_store_roundtrip() -> None:
    fake = _FakeS3Client()
    store = S3BlobStore(Settings(_env_file=None, s3_bucket="b"), client_fn=lambda: fake)

    store.put("documents/x", b"pdf-bytes")
    assert store.get("documents/x") == b"pdf-bytes"
    assert fake.store[("b", "documents/x")] == b"pdf-bytes"  # went to the right bucket/key
    store.delete("documents/x")
    assert ("b", "documents/x") not in fake.store
