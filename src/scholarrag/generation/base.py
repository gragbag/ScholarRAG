"""The generated answer type.

An :class:`Answer` is what the whole system exists to produce: the grounded,
cited text plus the source chunks the model actually cited (in citation order),
so a client can render ``[1]`` as a link back to the real document.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholarrag.retrieval.base import RetrievedChunk


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer and the sources it cited."""

    text: str  # the model's answer, with inline [n] citation markers
    sources: list[RetrievedChunk]  # the chunks cited, in order of first citation
