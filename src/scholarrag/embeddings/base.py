"""Embedder protocol and shared types.

An embedder turns text into a fixed-length vector positioned so that similar
meanings sit close together (see the vector-store docs for the geometry). Like
:class:`~scholarrag.vectorstore.VectorStore`, it's an interface with swappable
implementations — a real local model for production, a dependency-free fake for
tests/CI.

Note the *two* embed methods. Retrieval is asymmetric — a short question is
matched against longer passages — and models like BGE do better when the query
carries an instruction prefix that the passages don't. Splitting
``embed_query`` from ``embed_documents`` lets each implementation honour that.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# A single embedding: a list of floats of length ``Embedder.dim``.
Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Constructing one must not do heavy I/O."""

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors this embedder produces."""
        ...

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of passages/chunks (ingestion side)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a single search query (retrieval side)."""
        ...
