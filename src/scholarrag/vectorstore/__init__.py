"""Vector store abstraction.

Pinecone is a managed cloud service, not a local container, so it sits behind a
small :class:`VectorStore` protocol. Two implementations exist:

* :class:`PineconeVectorStore` — production / default.
* :class:`LocalVectorStore` — in-process, used automatically in tests and CI
  (and local dev) whenever no Pinecone key is configured.

This keeps CI free and deterministic and is itself a good design talking point.
Use :func:`build_vector_store` to get the right implementation from settings.
"""

from __future__ import annotations

from scholarrag.config import Settings, get_settings
from scholarrag.vectorstore.base import (
    QueryMatch,
    VectorRecord,
    VectorStore,
)
from scholarrag.vectorstore.local import LocalVectorStore

__all__ = [
    "LocalVectorStore",
    "QueryMatch",
    "VectorRecord",
    "VectorStore",
    "build_vector_store",
]


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the vector store implied by configuration.

    Falls back to :class:`LocalVectorStore` unless Pinecone is explicitly
    selected/available, so importing this never requires cloud credentials.
    """
    settings = settings or get_settings()
    if settings.use_pinecone:
        # Imported lazily so the pinecone SDK is only needed when actually used.
        from scholarrag.vectorstore.pinecone import PineconeVectorStore

        return PineconeVectorStore(settings)
    return LocalVectorStore(dim=settings.embedding_dim)
