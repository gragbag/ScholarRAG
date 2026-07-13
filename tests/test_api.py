"""API smoke tests: health, readiness, info, and correlation-id propagation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from scholarrag import __version__
from scholarrag.api.middleware import CORRELATION_HEADER


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


def test_ready(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["checks"]["vector_store"] is True


def test_info_reports_resolved_config(client: TestClient) -> None:
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_provider"] == "anthropic"
    assert body["llm_model_cheap"] == "claude-haiku-4-5"
    assert body["llm_model_strong"] == "claude-sonnet-5"
    assert body["vector_store"] == "local"  # no Pinecone key in tests
    assert body["corpus_profile"] == "research_papers"
    assert "research_papers" in body["corpus_profiles_available"]


def test_correlation_id_generated_and_echoed(client: TestClient) -> None:
    resp = client.get("/health")
    assert CORRELATION_HEADER in resp.headers
    assert resp.headers[CORRELATION_HEADER]


def test_correlation_id_honours_inbound_header(client: TestClient) -> None:
    resp = client.get("/health", headers={CORRELATION_HEADER: "trace-abc-123"})
    assert resp.headers[CORRELATION_HEADER] == "trace-abc-123"
