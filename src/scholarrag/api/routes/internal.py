"""Internal ingestion endpoint — the Cloud Tasks push target.

When ``QUEUE_BACKEND=cloudtasks``, Cloud Tasks POSTs each job here and this runs
the same :func:`run_ingestion` the Celery worker would. Protected by a shared
secret (``INTERNAL_SECRET``). Retry is the QUEUE's job: a transient failure
returns 5xx so Cloud Tasks retries; on the final attempt we dead-letter so a
poison document doesn't retry forever.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from scholarrag.config import get_settings
from scholarrag.db import repository as repo
from scholarrag.db.engine import session_scope
from scholarrag.db.models import IngestionStatus
from scholarrag.ingestion.runner import run_ingestion

router = APIRouter(prefix="/internal", tags=["internal"])


class IngestTask(BaseModel):
    document_id: str


@router.post("/ingest", status_code=status.HTTP_204_NO_CONTENT)
async def internal_ingest(
    body: IngestTask,
    x_internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
    retry_count: int = Header(default=0, alias="X-CloudTasks-TaskRetryCount"),
) -> None:
    """Process one document (Cloud Tasks push target). 401 unless the shared
    secret matches; 503 on a transient failure so the queue retries."""
    settings = get_settings()
    if not settings.internal_secret or x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

    document_id = uuid.UUID(body.document_id)
    try:
        # Permanent failures are dead-lettered inside run_ingestion; only transient
        # ones raise — those we hand back to Cloud Tasks to retry.
        run_ingestion(document_id)
    except Exception as exc:
        if retry_count + 1 >= settings.cloud_tasks_max_attempts:
            # Last delivery attempt — stop retrying, dead-letter the document.
            with session_scope() as session:
                repo.set_document_status(
                    session, document_id, IngestionStatus.dead_letter, error=str(exc)
                )
                session.commit()
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="transient; retry"
        ) from exc
