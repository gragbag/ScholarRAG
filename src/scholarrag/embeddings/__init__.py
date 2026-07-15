"""Embeddings package.

Text -> vectors, behind an :class:`Embedder` protocol with swappable backends:

* :class:`LocalEmbedder`  — sentence-transformers BGE (default / production).
* :class:`FakeEmbedder`   — deterministic, dependency-free (tests / CI).

Use :func:`build_embedder` to get the implementation implied by settings.
"""

from __future__ import annotations

from scholarrag.config import Settings, get_settings
from scholarrag.embeddings.base import Embedder, Vector
from scholarrag.embeddings.fake import FakeEmbedder
from scholarrag.embeddings.local import BGE_QUERY_PREFIX, LocalEmbedder

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "LocalEmbedder",
    "Vector",
    "build_embedder",
]


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Return the embedder implied by configuration (``EMBEDDING_PROVIDER``)."""
    settings = settings or get_settings()
    provider = settings.embedding_provider
    if provider == "fake":
        return FakeEmbedder(dim=settings.embedding_dim)
    if provider == "local":
        return LocalEmbedder(
            settings.embedding_model,
            dim=settings.embedding_dim,
            query_prefix=BGE_QUERY_PREFIX,
        )
    # e.g. "openai" — wired in a later phase.
    raise ValueError(f"unsupported embedding provider: {provider!r}")
