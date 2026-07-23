"""Grounded generation — turn retrieved chunks into a cited answer.

The ``Answerer`` is the "G" in RAG. It lays the chunks out as numbered sources,
asks the *strong* LLM tier to answer using only those sources (with [n]
citations), and returns the answer plus the chunks it actually cited.
"""

from __future__ import annotations

from collections.abc import Iterator

from scholarrag.generation.base import Answer
from scholarrag.generation.citations import extract_citations
from scholarrag.generation.prompts import GROUNDED_SYSTEM, render_answer_prompt
from scholarrag.guardrails.output import enforce_grounding
from scholarrag.llm.base import LLMClient
from scholarrag.retrieval.base import RetrievedChunk


def cited_sources(text: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Map the ``[n]`` markers in ``text`` back to their chunks (out-of-range ignored)."""
    return [chunks[n - 1] for n in extract_citations(text) if 1 <= n <= len(chunks)]


class Answerer:
    """Generates a grounded, cited :class:`Answer` from retrieved chunks."""

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def answer(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        "Answer ``query`` grounded in ``chunks``, returning text + cited sources."

        prompt = render_answer_prompt(query, chunks)
        text = self._llm.complete(prompt, system=GROUNDED_SYSTEM, tier="strong")
        cited = extract_citations(text)

        sources = [chunks[n - 1] for n in cited if 1 <= n <= len(chunks)]

        return enforce_grounding(Answer(text=text, sources=sources))

    def answer_stream(self, query: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """Stream the grounded answer as text deltas (citations resolved by the caller)."""
        prompt = render_answer_prompt(query, chunks)
        yield from self._llm.stream(prompt, system=GROUNDED_SYSTEM, tier="strong")
