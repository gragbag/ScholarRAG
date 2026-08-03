"""Enqueuer backends — how a registered document is handed off for processing.

The API depends only on the ``Enqueuer`` protocol (``Callable[[UUID], None]``);
:func:`scholarrag.api.deps.get_enqueuer` picks the concrete one from
``QUEUE_BACKEND``. All three ultimately run :func:`run_ingestion`:

* ``celery``     — enqueue a Celery task (in ``deps.enqueue_ingestion``).
* ``eager``      — process in a background thread of the API process (this file).
* ``cloudtasks`` — push an HTTP task to Cloud Tasks -> POST /internal/ingest.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from typing import Any

from scholarrag.config import Settings
from scholarrag.db import repository as repo
from scholarrag.db.engine import session_scope
from scholarrag.db.models import IngestionStatus
from scholarrag.ingestion.runner import run_ingestion


def _process_eager(document_id: uuid.UUID) -> None:
    """Run ingestion inline. There's no queue to retry, so a transient failure is
    recorded as ``failed`` (rather than left mid-flight); permanent failures are
    already dead-lettered by ``run_ingestion``."""
    try:
        run_ingestion(document_id)
    except Exception as exc:
        with session_scope() as session:
            repo.set_document_status(session, document_id, IngestionStatus.failed, error=str(exc))
            session.commit()


class EagerEnqueuer:
    """Process the document in a background thread of the API process — NO separate
    worker. Returns immediately (like a real queue would). Handy for local dev."""

    def __call__(self, document_id: uuid.UUID) -> None:
        threading.Thread(target=_process_eager, args=(document_id,), daemon=True).start()


# Test seam: create_task_fn(queue=(project, location, queue), task=<dict>) -> None.
CreateTaskFn = Callable[..., Any]


class CloudTasksEnqueuer:
    """Push an HTTP task to Google Cloud Tasks that POSTs to ``/internal/ingest``.

    The queue owns delivery + retries; the route does the work. The
    ``google-cloud-tasks`` client is lazy-imported (deploy-only) and
    ``create_task_fn`` is a test seam, so this is exercised without GCP.
    """

    def __init__(self, settings: Settings, *, create_task_fn: CreateTaskFn | None = None) -> None:
        if settings.internal_ingest_url is None and create_task_fn is None:
            raise ValueError("INTERNAL_INGEST_URL is required for CloudTasksEnqueuer")
        self._settings = settings
        self._create_task_fn = create_task_fn
        self._client: Any = None

    def _client_lazy(self) -> Any:
        if self._client is None:
            from google.cloud import tasks_v2

            self._client = tasks_v2.CloudTasksClient()
        return self._client

    def _task(self, document_id: uuid.UUID) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._settings.internal_secret is not None:
            headers["X-Internal-Secret"] = self._settings.internal_secret
        return {
            "http_request": {
                "http_method": "POST",  # proto-plus accepts the enum name as a string
                "url": self._settings.internal_ingest_url,
                "headers": headers,
                "body": json.dumps({"document_id": str(document_id)}).encode(),
            }
        }

    def __call__(self, document_id: uuid.UUID) -> None:
        s = self._settings
        task = self._task(document_id)
        if self._create_task_fn is not None:
            self._create_task_fn(
                queue=(s.gcp_project, s.gcp_location, s.cloud_tasks_queue), task=task
            )
            return
        client = self._client_lazy()
        parent = client.queue_path(s.gcp_project, s.gcp_location, s.cloud_tasks_queue)
        client.create_task(parent=parent, task=task)
