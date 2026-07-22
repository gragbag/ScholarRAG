"""Observability tests.

The no-op tests pass now: with no Langfuse keys configured, the tracing layer
must be invisible — decorated functions behave identically and helpers never
raise. The two skipped tests are the Step 1 exercise targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scholarrag.config import Settings
from scholarrag.observability import (
    configure_observability,
    configure_otel,
    flush,
    get_tracer,
    is_enabled,
    is_otel_enabled,
    observe,
    update_current_generation,
)


def test_disabled_by_default() -> None:
    assert is_enabled() is False


def test_configure_without_keys_stays_disabled() -> None:
    configure_observability(Settings(_env_file=None, langfuse_public_key=None))
    assert is_enabled() is False


def test_observe_is_transparent_when_disabled() -> None:
    @observe(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == "add"  # functools.wraps preserved identity


def test_helpers_are_noops_when_disabled() -> None:
    update_current_generation(model="m", usage={"input": 1, "output": 2})
    flush()  # neither should raise


# ── Exercise A — trace the pipeline (QueryEngine spans) ──────────────────────
def test_pipeline_stages_are_traced() -> None:
    from scholarrag.pipeline import QueryEngine

    # Our observe() uses functools.wraps, so decorated methods carry __wrapped__.
    assert hasattr(QueryEngine.query, "__wrapped__")
    assert hasattr(QueryEngine.answer_with_context, "__wrapped__")
    assert hasattr(QueryEngine._retrieve, "__wrapped__")


# ── Exercise B — log the Gemini call as a generation with usage ──────────────
@dataclass
class _Usage:
    prompt_token_count: int | None
    candidates_token_count: int | None


@dataclass
class _Response:
    text: str | None
    usage_metadata: _Usage | None


def test_gemini_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from scholarrag.llm import GeminiLLM

    recorded: dict[str, object] = {}

    def record(**kwargs: object) -> None:
        recorded.update(kwargs)

    # Patch the helper *in the gemini module's namespace* (requires the import
    # the exercise asks you to add).
    monkeypatch.setattr("scholarrag.llm.gemini.update_current_generation", record)

    llm = GeminiLLM(
        Settings(_env_file=None, gemini_model_cheap="flash-lite-x"),
        generate_fn=lambda **kw: _Response("an answer", _Usage(120, 45)),
    )
    assert llm.complete("Q", tier="cheap") == "an answer"

    assert recorded["model"] == "flash-lite-x"
    assert recorded["usage"] == {"input": 120, "output": 45}
    assert recorded["output"] == "an answer"


# ── OTel: no-op safety (pass now) ────────────────────────────────────────────
def test_otel_disabled_by_default() -> None:
    assert is_otel_enabled() is False


def test_configure_otel_without_endpoint_stays_disabled() -> None:
    configure_otel(Settings(_env_file=None, otel_exporter_endpoint=None))
    assert is_otel_enabled() is False


def test_get_tracer_is_noop_when_disabled() -> None:
    tracer = get_tracer("test")
    with tracer.start_as_current_span("anything") as span:
        span.set_attribute("k", "v")  # must absorb calls without the SDK


# ── OTel exercise A — the setup ritual ───────────────────────────────────────
def test_configure_otel_enables_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry")  # observability extra (absent in CI)
    from scholarrag.observability import otel

    try:
        # No Jaeger needed: the batch exporter sends in the background and
        # drops silently if nothing is listening.
        configure_otel(Settings(_env_file=None, otel_exporter_endpoint="http://localhost:4318"))
        assert is_otel_enabled() is True
        # The real tracer produces recording spans now.
        tracer = get_tracer("test")
        with tracer.start_as_current_span("probe") as span:
            span.set_attribute("k", "v")
    finally:
        otel._enabled = False  # don't leak enabled state into other tests


# ── OTel exercise B — manual spans in the hybrid retriever ───────────────────
def test_hybrid_emits_manual_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from sqlalchemy.orm import Session

    from scholarrag.retrieval import HybridRetriever, RetrievedChunk

    # A private provider exporting to memory — no global state, no network.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Patch the helper in hybrid's namespace (requires the import the exercise
    # asks you to add).
    monkeypatch.setattr(
        "scholarrag.retrieval.hybrid.get_tracer", lambda name: provider.get_tracer(name)
    )

    class _Stub:
        def retrieve(
            self, session: Session, query: str, *, top_k: int = 10
        ) -> list[RetrievedChunk]:
            return []

    HybridRetriever(dense=_Stub(), lexical=_Stub()).retrieve(Session(), "q", top_k=3)

    names = {span.name for span in exporter.get_finished_spans()}
    assert "retrieve.dense" in names
    assert "retrieve.lexical" in names
