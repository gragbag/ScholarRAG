"""S3-compatible blob store — Cloudflare R2, AWS S3, or MinIO.

All three speak the S3 API, so one boto3 client targets any of them; the
``endpoint_url`` decides which:

    R2     endpoint=https://<account-id>.r2.cloudflarestorage.com  region="auto"
    AWS S3 endpoint=None (default)                                 region="us-east-1"
    MinIO  endpoint=http://localhost:9000

boto3 is lazy-imported (deploy-only, the ``s3`` extra); ``client_fn`` is a test
seam so this is exercised without boto3, credentials, or a bucket.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scholarrag.blobstore.base import BlobNotFoundError
from scholarrag.config import Settings

# Test seam: returns a boto3-S3-like client (put_object/get_object/delete_object).
ClientFn = Callable[[], Any]


class S3BlobStore:
    """S3-compatible :class:`BlobStore`. Construction does no I/O (lazy client)."""

    def __init__(self, settings: Settings, *, client_fn: ClientFn | None = None) -> None:
        self._settings = settings
        self._client_fn = client_fn
        self._client: Any = None

    def _client_lazy(self) -> Any:
        if self._client is None:
            if self._client_fn is not None:
                self._client = self._client_fn()
            else:
                import boto3

                self._client = boto3.client(
                    "s3",
                    endpoint_url=self._settings.s3_endpoint_url,
                    aws_access_key_id=self._settings.s3_access_key_id,
                    aws_secret_access_key=self._settings.s3_secret_access_key,
                    region_name=self._settings.s3_region,
                )
        return self._client

    def put(self, key: str, data: bytes) -> None:
        "Upload ``data`` to ``key``."
        self._client_lazy().put_object(
            Bucket=self._settings.s3_bucket,
            Key=key,
            Body=data,
        )

    def get(self, key: str) -> bytes:
        "Download and return the bytes at ``key``.="
        try:
            resp = self._client_lazy().get_object(Bucket=self._settings.s3_bucket, Key=key)
        except self._client_lazy().exceptions.NoSuchKey as exc:
            raise BlobNotFoundError(key) from exc

        result: bytes = resp["Body"].read()
        return result

    def delete(self, key: str) -> None:
        "Delete ``key`` (S3 delete is a no-op if it's already gone)."
        self._client_lazy().delete_object(Bucket=self._settings.s3_bucket, Key=key)
