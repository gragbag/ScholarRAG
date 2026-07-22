"""OpenTelemetry app tracing — spans for HTTP, SQL, and Celery, shipped to Jaeger.

Complements Langfuse: Langfuse understands the LLM layer (prompts/tokens/cost),
OTel sees the application around it (FastAPI requests, SQLAlchemy queries, the
Celery hop from API to worker). Same gating philosophy as ``langfuse.py``:

* package not installed        -> everything no-ops (CI untouched)
* ``OTEL_EXPORTER_ENDPOINT`` unset -> everything no-ops (the default)
* endpoint set                 -> spans flow to Jaeger via OTLP/HTTP

``configure_otel`` is called at API startup (with the FastAPI app) and at Celery
worker startup (without); ``get_tracer`` hands out a real or no-op tracer for
manual spans (exercise B in the hybrid retriever).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from fastapi import FastAPI

_enabled = False


def is_otel_enabled() -> bool:
    """Whether OTel tracing is active (endpoint configured and SDK importable)."""
    return _enabled


class _NoOpSpan:
    """Absorbs span calls when tracing is off."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None


class _NoOpTracer:
    """Hand-rolled stand-in so callers never need the SDK installed."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()


_NOOP_TRACER = _NoOpTracer()


def get_tracer(name: str) -> Any:
    """Return a tracer for manual spans — the real one when enabled, else a no-op.

    Call this *at call time* (inside the function you're tracing), not at module
    import — tracing is configured at startup, after imports.
    """
    if not _enabled:
        return _NOOP_TRACER
    from opentelemetry import trace

    return trace.get_tracer(name)


def configure_otel(settings: Settings, app: FastAPI | None = None) -> None:
    """Enable OTel tracing if an exporter endpoint is configured; else no-op."""
    global _enabled
    if not settings.otel_exporter_endpoint:
        return
    try:
        import opentelemetry.sdk  # noqa: F401  (observability extra present?)
    except ImportError:
        return
    _setup_tracing(settings, app)
    _enabled = True


def _setup_tracing(settings: Settings, app: FastAPI | None) -> None:
    "Build the provider/exporter and switch on auto-instrumentation."
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    if app is not None:  # API process only; the Celery worker has no app
        FastAPIInstrumentor.instrument_app(app)
    from scholarrag.db.engine import get_engine

    SQLAlchemyInstrumentor().instrument(engine=get_engine())
    CeleryInstrumentor().instrument()  # producer AND worker side
