"""Grounded-generation tests (Exercises A & C — hermetic, use FakeLLM).

``format_sources`` passes now. Remove each ``@pytest.mark.skip`` once you've
implemented the target function.
"""

from __future__ import annotations

import uuid

from scholarrag.generation import Answerer, cited_sources, extract_citations
from scholarrag.generation.prompts import format_sources
from scholarrag.llm import FakeLLM
from scholarrag.retrieval import RetrievedChunk


def _chunk(cid: str, text: str, filename: str = "a.txt") -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, document_id=uuid.uuid4(), chunk_index=0, text=text, filename=filename, score=0.0
    )


def test_format_sources_numbers_chunks() -> None:
    # Format-agnostic (survives the Step 3 injection-hardening rewrite): each
    # chunk's number, filename, and text must appear, in order.
    out = format_sources([_chunk("0", "alpha", "one.md"), _chunk("1", "beta", "two.md")])
    for fragment in ("1", "one.md", "alpha", "2", "two.md", "beta"):
        assert fragment in out
    assert out.index("alpha") < out.index("beta")


# ── Exercise C — extract_citations ───────────────────────────────────────────
def test_extract_citations() -> None:
    assert extract_citations("grounds it [1] and also [3], see [1] again") == [1, 3]
    assert extract_citations("no citations here") == []


# ── Exercise A — Answerer.answer ─────────────────────────────────────────────
def test_answerer_returns_cited_sources() -> None:
    chunks = [
        _chunk("0", "RAG grounds answers"),
        _chunk("1", "dense is semantic"),
        _chunk("2", "x"),
    ]
    # The model cites sources 1 and 2, but not 3.
    llm = FakeLLM(["RAG grounds answers [1] using dense retrieval [2]."])
    answer = Answerer(llm=llm).answer("how does RAG work", chunks)

    assert answer.text == "RAG grounds answers [1] using dense retrieval [2]."
    assert [s.id for s in answer.sources] == ["0", "1"]  # cited chunks only, in order
    assert llm.calls[0]["tier"] == "strong"  # generation uses the strong tier


# ── Step 4b streaming (scaffolded) ───────────────────────────────────────────
def test_cited_sources_maps_and_ignores_out_of_range() -> None:
    chunks = [_chunk("0", "a"), _chunk("1", "b")]
    # [1] and [2] map to chunks; [9] is out of range and ignored.
    assert [c.id for c in cited_sources("uses [1], [2], and [9]", chunks)] == ["0", "1"]


def test_answer_stream_yields_the_answer_text() -> None:
    chunks = [_chunk("0", "RAG grounds answers")]
    llm = FakeLLM(["RAG grounds answers [1]."])
    tokens = list(Answerer(llm=llm).answer_stream("q", chunks))
    assert "".join(tokens) == "RAG grounds answers [1]."
    assert llm.calls[0]["tier"] == "strong"
