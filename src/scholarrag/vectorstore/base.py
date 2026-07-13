"""VectorStore protocol and shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Metadata values we allow on a record. Kept JSON-serialisable so the same
# shape works for the in-memory store and for Pinecone.
MetadataValue = str | int | float | bool | list[str]
Metadata = dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A single vector to upsert: a unique id, its embedding, and metadata."""

    id: str
    values: list[float]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryMatch:
    """A single nearest-neighbour result."""

    id: str
    score: float
    metadata: Metadata = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector store surface: upsert, query, delete.

    Implementations must be safe to construct without any network I/O; the
    actual backend connection may be established lazily.
    """

    def upsert(self, records: list[VectorRecord], *, namespace: str = "") -> int:
        """Insert or replace ``records``. Returns the number written."""
        ...

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter: Metadata | None = None,
    ) -> list[QueryMatch]:
        """Return up to ``top_k`` nearest neighbours to ``vector``."""
        ...

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        namespace: str = "",
        delete_all: bool = False,
    ) -> int:
        """Delete records by id (or everything in a namespace). Returns count."""
        ...

    def count(self, *, namespace: str = "") -> int:
        """Return the number of stored vectors in ``namespace``."""
        ...
