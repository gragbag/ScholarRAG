"""Queue-backend tests — enqueuer selection + the eager/Cloud Tasks backends.

None need a broker, GCP, or a DB: Cloud Tasks uses its create_task seam, eager's
runner is monkeypatched, and the route test stops at the auth check.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from scholarrag.api import deps
from scholarrag.api.main import create_app
from scholarrag.config import Settings
from scholarrag.ingestion.enqueue import CloudTasksEnqueuer, EagerEnqueuer, _process_eager


def test_cloudtasks_enqueuer_builds_http_task() -> None:
    captured: dict[str, object] = {}

    def fake_create(*, queue: object, task: object) -> None:
        captured["queue"] = queue
        captured["task"] = task

    settings = Settings(
        _env_file=None,
        internal_ingest_url="https://svc.run.app/internal/ingest",
        internal_secret="sekret",
        gcp_project="proj",
        gcp_location="us-central1",
        cloud_tasks_queue="ingestion",
    )
    enqueuer = CloudTasksEnqueuer(settings, create_task_fn=fake_create)
    doc = uuid.uuid4()
    enqueuer(doc)

    assert captured["queue"] == ("proj", "us-central1", "ingestion")
    req = captured["task"]["http_request"]  # type: ignore[index]
    assert req["http_method"] == "POST"
    assert req["url"] == "https://svc.run.app/internal/ingest"
    assert req["headers"]["X-Internal-Secret"] == "sekret"
    assert json.loads(req["body"]) == {"document_id": str(doc)}


def test_cloudtasks_enqueuer_requires_url() -> None:
    with pytest.raises(ValueError, match="INTERNAL_INGEST_URL"):
        CloudTasksEnqueuer(Settings(_env_file=None, internal_ingest_url=None))


def test_eager_process_runs_the_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, uuid.UUID] = {}
    monkeypatch.setattr(
        "scholarrag.ingestion.enqueue.run_ingestion",
        lambda document_id: seen.__setitem__("id", document_id),
    )
    doc = uuid.uuid4()
    _process_eager(doc)
    assert seen["id"] == doc


def test_get_enqueuer_selects_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def with_backend(backend: str) -> object:
        monkeypatch.setattr(
            deps,
            "get_settings",
            lambda: Settings(
                _env_file=None,
                queue_backend=backend,
                internal_ingest_url="https://x/internal/ingest",
            ),
        )
        return deps.get_enqueuer()

    assert isinstance(with_backend("eager"), EagerEnqueuer)
    assert isinstance(with_backend("cloudtasks"), CloudTasksEnqueuer)
    assert with_backend("celery") is deps.enqueue_ingestion


def test_internal_ingest_rejects_bad_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    import scholarrag.api.routes.internal as internal_mod

    monkeypatch.setattr(
        internal_mod, "get_settings", lambda: Settings(_env_file=None, internal_secret="s3cret")
    )
    with TestClient(create_app(Settings(_env_file=None))) as client:
        doc = {"document_id": str(uuid.uuid4())}
        assert client.post("/internal/ingest", json=doc).status_code == 401  # missing header
        bad = {"X-Internal-Secret": "wrong"}
        assert client.post("/internal/ingest", json=doc, headers=bad).status_code == 401
