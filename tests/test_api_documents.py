"""Document API tests.

Hermetic: we override the API's dependencies with the isolated test DB session, a
FakeEmbedder-backed pipeline, and a *spy* enqueuer — so no Redis, no worker, no
model. The two skipped tests are the Step 6 exercises.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.orm import Session

from scholarrag.api.deps import get_db, get_enqueuer, get_pipeline
from scholarrag.api.main import create_app
from scholarrag.config import Settings
from scholarrag.corpus import get_corpus_profile
from scholarrag.db.models import IngestionStatus
from scholarrag.embeddings import FakeEmbedder
from scholarrag.ingestion import IngestionPipeline
from scholarrag.vectorstore import LocalVectorStore

DIM = 32


@pytest.fixture
def api(db: Session, settings: Settings) -> Iterator[tuple[TestClient, list[uuid.UUID]]]:
    app = create_app(settings)
    fake_pipeline = IngestionPipeline(
        embedder=FakeEmbedder(dim=DIM), vector_store=LocalVectorStore(dim=DIM)
    )
    enqueued: list[uuid.UUID] = []
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[get_enqueuer] = lambda: enqueued.append
    with TestClient(app) as client:
        yield client, enqueued


def _upload(client: TestClient, name: str, content: bytes) -> Response:
    resp: Response = client.post("/documents", files={"file": (name, content, "text/plain")})
    return resp


def test_upload_returns_202_and_enqueues(api: tuple[TestClient, list[uuid.UUID]]) -> None:
    client, enqueued = api
    resp = _upload(client, "paper.txt", b"retrieval augmented generation " * 50)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["skipped"] is False
    assert len(enqueued) == 1  # handed off to a worker exactly once


def test_upload_unsupported_type_returns_415(api: tuple[TestClient, list[uuid.UUID]]) -> None:
    client, enqueued = api
    resp = client.post(
        "/documents", files={"file": ("weird.xyz", b"data", "application/octet-stream")}
    )
    assert resp.status_code == 415
    assert enqueued == []


def test_upload_duplicate_is_skipped(api: tuple[TestClient, list[uuid.UUID]]) -> None:
    client, enqueued = api
    content = b"identical bytes " * 50
    first = _upload(client, "a.txt", content)
    second = _upload(client, "a.txt", content)
    assert first.json()["skipped"] is False
    assert second.json()["skipped"] is True
    assert len(enqueued) == 1  # not re-enqueued


def test_list_documents(api: tuple[TestClient, list[uuid.UUID]]) -> None:
    client, _ = api
    _upload(client, "a.txt", b"content one " * 50)
    _upload(client, "b.txt", b"content two " * 50)
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


# ── Step 6 exercise A — implement GET /documents/{id} ───────────────────────
def test_get_document_status(api: tuple[TestClient, list[uuid.UUID]]) -> None:
    client, _ = api
    up = _upload(client, "paper.txt", b"some content " * 50).json()
    resp = client.get(f"/documents/{up['document_id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == up["document_id"]
    assert resp.json()["filename"] == "paper.txt"
    # unknown id → 404
    assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404


# ── Step 6 exercise B — implement scripts.seed.ingest_corpus ────────────────
def test_ingest_corpus(db: Session, tmp_path: Path) -> None:
    from scholarrag.scripts.seed import ingest_corpus

    (tmp_path / "a.txt").write_text("hello world " * 50)
    (tmp_path / "b.md").write_text("# doc\n\nsome content " * 50)
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01\x02")  # unsupported → skipped

    pipe = IngestionPipeline(embedder=FakeEmbedder(dim=DIM), vector_store=LocalVectorStore(dim=DIM))
    results = ingest_corpus(
        pipe, db, corpus_dir=tmp_path, profile=get_corpus_profile("generic_docs")
    )

    assert len(results) == 2  # the .bin file was skipped
    assert all(r.status is IngestionStatus.completed for r in results)
