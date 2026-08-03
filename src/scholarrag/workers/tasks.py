"""The ingestion Celery task — a thin wrapper over the shared runner.

The actual work and the transient-vs-permanent failure policy live in
:mod:`scholarrag.ingestion.runner` (shared with the ``eager`` enqueuer and the
Cloud Tasks route). This module only adds Celery's *retry* mechanics on top:

* **transient** failures → retry with exponential backoff, up to ``max_retries``;
* **permanent** failures → the runner already marked the document ``dead_letter``.

(Redis has no native dead-letter queue like RabbitMQ, so we implement it at the
application level, on the document row.)
"""

from __future__ import annotations

import uuid

from celery import Task
from sqlalchemy.orm import Session

from scholarrag.db import repository as repo
from scholarrag.db.engine import session_scope
from scholarrag.db.models import IngestionStatus
from scholarrag.ingestion.runner import is_transient, run_ingestion
from scholarrag.workers.celery_app import app

# Re-exported so `from scholarrag.workers.tasks import is_transient` keeps working.
__all__ = ["ingest_document_task", "is_transient", "run_ingestion"]


def _record_dead_letter(session: Session, document_id: uuid.UUID, exc: BaseException) -> None:
    """Send a permanently-failed document to the dead-letter state."""
    repo.set_document_status(session, document_id, IngestionStatus.dead_letter, error=str(exc))
    session.commit()


@app.task(bind=True, acks_late=True, max_retries=5)  # type: ignore[untyped-decorator]
def ingest_document_task(self: Task, document_id: str) -> str:
    """Process a registered document in the background, with retry + DLQ."""
    doc_uuid = uuid.UUID(document_id)
    try:
        # run_ingestion dead-letters permanent failures itself and RAISES only
        # transient ones — which is exactly what we want to retry.
        run_ingestion(doc_uuid)
    except Exception as exc:
        if self.request.retries < self.max_retries:
            # Exponential backoff: 2s, 4s, 8s, ...
            raise self.retry(exc=exc, countdown=2**self.request.retries) from exc
        with session_scope() as session:
            _record_dead_letter(session, doc_uuid, exc)
        raise
    return str(doc_uuid)
