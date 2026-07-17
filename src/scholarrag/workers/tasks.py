"""The ingestion Celery task and its failure policy.

The task is a thin wrapper: it loads the process-wide pipeline, opens a DB
session, and calls ``pipeline.process(document_id)``. The interesting part is the
failure policy — deciding *retry* vs *dead-letter*:

* **transient** failures (network blip, timeout, a paused vector index) → retry
  with exponential backoff, up to ``max_retries``;
* **permanent** failures (a corrupt file, a bad content type) → don't retry;
  mark the document ``dead_letter`` and record the error.

(Redis has no native dead-letter queue like RabbitMQ, so we implement it here at
the application level, on the document row.)
"""

from __future__ import annotations

import uuid

from celery import Task
from sqlalchemy.orm import Session

from scholarrag.db import repository as repo
from scholarrag.db.engine import session_scope
from scholarrag.db.models import IngestionStatus
from scholarrag.ingestion import TransientIngestionError
from scholarrag.workers.celery_app import app
from scholarrag.workers.deps import get_pipeline

# Exceptions we treat as retryable. Anything else is permanent → dead-letter.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TransientIngestionError,
    ConnectionError,
    TimeoutError,
)


def is_transient(exc: BaseException) -> bool:
    "Whether ``exc`` is a *retryable* failure (True) or *permanent* (False)."
    return isinstance(exc, TRANSIENT_ERRORS)


def _record_dead_letter(session: Session, document_id: uuid.UUID, exc: BaseException) -> None:
    """Send a permanently-failed document to the dead-letter state."""
    repo.set_document_status(session, document_id, IngestionStatus.dead_letter, error=str(exc))
    session.commit()


@app.task(bind=True, acks_late=True, max_retries=5)  # type: ignore[untyped-decorator]
def ingest_document_task(self: Task, document_id: str) -> str:
    """Process a registered document in the background, with retry + DLQ."""
    pipeline = get_pipeline()
    doc_uuid = uuid.UUID(document_id)
    with session_scope() as session:
        try:
            result = pipeline.process(session, doc_uuid)
            return str(result.document_id)
        except Exception as exc:
            if is_transient(exc) and self.request.retries < self.max_retries:
                # Exponential backoff: 2s, 4s, 8s, ...
                raise self.retry(exc=exc, countdown=2**self.request.retries) from exc
            _record_dead_letter(session, doc_uuid, exc)
            raise
