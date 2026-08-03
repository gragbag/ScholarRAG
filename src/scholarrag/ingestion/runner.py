"""Run one document's ingestion — the shared core behind every queue backend.

The Celery task, the ``eager`` enqueuer, and (next) the Cloud Tasks
``/internal/ingest`` route all need to do the *same* thing: process a document,
and apply the transient-vs-permanent failure policy. That policy lives here once,
so no backend re-implements it — and this module imports NO Celery (the API/eager
paths must not pull the broker in).

Contract: on a **permanent** failure, mark the document ``dead_letter`` and
return normally; on a **transient** failure, RAISE — so the caller (Celery's
retry, Cloud Tasks' retry) can try again.
"""

from __future__ import annotations

import uuid

from scholarrag.db import repository as repo
from scholarrag.db.engine import session_scope
from scholarrag.db.models import IngestionStatus
from scholarrag.ingestion import TransientIngestionError
from scholarrag.workers.deps import get_pipeline

# Retryable failures. Anything else is permanent → dead-letter.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TransientIngestionError,
    ConnectionError,
    TimeoutError,
)


def is_transient(exc: BaseException) -> bool:
    "Whether ``exc`` is a *retryable* failure (True) or *permanent* (False)."
    return isinstance(exc, TRANSIENT_ERRORS)


def run_ingestion(document_id: uuid.UUID) -> None:
    """Process a document end-to-end.

    Permanent failure → mark ``dead_letter`` and return. Transient failure →
    RAISE, so the caller's retry mechanism (Celery / Cloud Tasks) can retry.
    """
    pipeline = get_pipeline()
    with session_scope() as session:
        try:
            pipeline.process(session, document_id)
        except Exception as exc:
            if is_transient(exc):
                raise
            repo.set_document_status(
                session, document_id, IngestionStatus.dead_letter, error=str(exc)
            )
            session.commit()
