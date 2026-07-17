"""Worker/task tests.

``test_task_is_registered`` passes now (no broker needed). The ``is_transient``
tests are the Step 5 exercise — remove the ``@pytest.mark.skip`` once you've
implemented the retry-vs-dead-letter classifier.
"""

from __future__ import annotations

from scholarrag.ingestion import TransientIngestionError, UnsupportedFileTypeError
from scholarrag.workers.tasks import ingest_document_task, is_transient


def test_task_is_registered() -> None:
    # The Celery task exists under its expected name (so the API can enqueue it).
    assert ingest_document_task.name == "scholarrag.workers.tasks.ingest_document_task"


def test_is_transient_true_for_retryable() -> None:
    assert is_transient(TransientIngestionError("network blip")) is True
    assert is_transient(ConnectionError()) is True
    assert is_transient(TimeoutError()) is True


def test_is_transient_false_for_permanent() -> None:
    assert is_transient(UnsupportedFileTypeError("bad type")) is False
    assert is_transient(ValueError("corrupt document")) is False
    assert is_transient(Exception("some bug")) is False
