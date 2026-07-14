"""Hands-on exercises — your turn.

These are learning exercises with pre-written *failing* tests as targets. Each
is skipped so the suite stays green until you start. Workflow for each:

  1. Remove the ``@pytest.mark.skip(...)`` line above the test.
  2. Run ``make test`` and watch it fail (red).
  3. Implement the feature until the test passes (green).

Full instructions, hints, and acceptance criteria live in ``EXERCISES.md``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from scholarrag.corpus import available_profiles, get_corpus_profile


# ---------------------------------------------------------------------------
# Exercise 1 — add a `legal_docs` corpus profile (edit src/scholarrag/corpus.py)
# ---------------------------------------------------------------------------
# @pytest.mark.skip(reason="Exercise 1 — see EXERCISES.md; delete this skip to start.")
def test_legal_docs_profile_registered() -> None:
    assert "legal_docs" in available_profiles()
    profile = get_corpus_profile("legal_docs")
    assert profile.name == "legal_docs"
    # A legal corpus should still accept PDFs and have a non-empty prompt.
    assert ".pdf" in profile.file_types
    assert profile.answer_system_prompt


# ---------------------------------------------------------------------------
# Exercise 2 — add GET /corpus/{name} (edit src/scholarrag/api/main.py)
# ---------------------------------------------------------------------------
# @pytest.mark.skip(reason="Exercise 2 — see EXERCISES.md; delete this skip to start.")
def test_get_corpus_endpoint(client: TestClient) -> None:
    resp = client.get("/corpus/research_papers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "research_papers"
    assert body["chunk_size"] > 0
    assert ".pdf" in body["file_types"]


# @pytest.mark.skip(reason="Exercise 2 (stretch) — see EXERCISES.md; delete this skip to start.")
def test_get_corpus_unknown_returns_404(client: TestClient) -> None:
    resp = client.get("/corpus/does_not_exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Exercise 3 — add a `fetch(id)` method to the VectorStore protocol.
#
# This one is NOT pre-written here on purpose: the test calls `store.fetch(...)`,
# and mypy would fail to type-check a method that doesn't exist yet. Add the
# method first (that's the lesson — see EXERCISES.md), then paste the test from
# EXERCISES.md into tests/test_vectorstore.py.
# ---------------------------------------------------------------------------
