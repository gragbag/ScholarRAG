"""The Celery application.

Celery needs a *broker* (the message transport) to hand tasks from producers
(the API) to consumers (workers). We use Redis. ``task_acks_late`` means a task
is only acknowledged after it finishes, so a crashed worker's task is redelivered
— at-least-once delivery, which is safe precisely because ingestion is
idempotent by content hash.

Run a worker with:
    celery -A scholarrag.workers.celery_app worker -l info
"""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init

from scholarrag.config import get_settings


def create_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "scholarrag",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Reliability: ack after completion, redeliver on worker loss.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
    )
    # Discovers scholarrag.workers.tasks so the tasks register with this app.
    app.autodiscover_tasks(["scholarrag.workers"])
    return app


app = create_celery()


@worker_process_init.connect  # type: ignore[untyped-decorator]  # celery's is untyped
def _init_observability(**_kwargs: object) -> None:
    """Configure tracing in each worker process (the API configures its own).

    With OTel's Celery instrumentation active on both sides, a trace started by
    an upload request continues into the worker that ingests the document —
    distributed tracing across the API -> Redis -> worker hop.
    """
    from scholarrag.observability import configure_observability, configure_otel

    settings = get_settings()
    configure_observability(settings)
    configure_otel(settings)  # no FastAPI app in a worker
