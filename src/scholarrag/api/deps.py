"""FastAPI dependencies for the API routes.

These are the injection seams: routes declare what they need via ``Depends(...)``,
and tests override them (a test DB session, a fake pipeline, a spy enqueuer) so
endpoint tests need no Postgres, no Redis, and no model download.

``get_enqueuer`` is the important one — in production it returns a function that
pushes the Celery task onto Redis; in tests it's overridden with a no-op spy, so
we can assert "the route enqueued" without a running broker.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from functools import lru_cache

from sqlalchemy.orm import Session

from scholarrag.db.engine import session_scope
from scholarrag.ingestion import IngestionPipeline
from scholarrag.pipeline import QueryEngine, build_query_engine
from scholarrag.workers.deps import get_pipeline as _get_worker_pipeline
from scholarrag.workers.tasks import ingest_document_task

# A function that hands a document off to a background worker.
Enqueuer = Callable[[uuid.UUID], None]


def get_db() -> Iterator[Session]:
    """Yield a transactional session for the request (commit/rollback at the edges)."""
    with session_scope() as session:
        yield session


def get_pipeline() -> IngestionPipeline:
    """Return the process-wide ingestion pipeline (built once, from settings)."""
    return _get_worker_pipeline()


def enqueue_ingestion(document_id: uuid.UUID) -> None:
    """Push the ingestion task onto the queue (production behaviour)."""
    ingest_document_task.delay(str(document_id))


def get_enqueuer() -> Enqueuer:
    """Return the enqueue function (overridden with a spy in tests)."""
    return enqueue_ingestion


@lru_cache
def _cached_query_engine() -> QueryEngine:
    """Build the query engine once (it holds long-lived retriever + LLM client)."""
    return build_query_engine()


def get_query_engine() -> QueryEngine:
    """Return the process-wide RAG query engine (overridden with a fake in tests)."""
    return _cached_query_engine()
