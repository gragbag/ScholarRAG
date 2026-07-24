"""Observability package — Langfuse LLM tracing (Phase 4 Step 1).

Everything here is a safe no-op unless Langfuse keys are configured AND the
``observability`` extra is installed, so the core pipeline, tests, and CI are
untouched. OpenTelemetry app-tracing joins this package later in Step 1.
"""

from __future__ import annotations

from scholarrag.observability.langfuse import (
    configure_observability,
    flush,
    get_langchain_callbacks,
    is_enabled,
    observe,
    update_current_generation,
    update_current_trace,
)
from scholarrag.observability.otel import configure_otel, get_tracer, is_otel_enabled

__all__ = [
    "configure_observability",
    "configure_otel",
    "flush",
    "get_langchain_callbacks",
    "get_tracer",
    "is_enabled",
    "is_otel_enabled",
    "observe",
    "update_current_generation",
    "update_current_trace",
]
