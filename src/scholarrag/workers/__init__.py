"""Async ingestion workers (Celery + Redis).

The API enqueues ``ingest_document_task(document_id)``; a worker process consumes
it and runs the ingestion pipeline in the background.
"""

from __future__ import annotations

from scholarrag.workers.celery_app import app
from scholarrag.workers.tasks import ingest_document_task

__all__ = ["app", "ingest_document_task"]
