"""Relational persistence layer (PostgreSQL via SQLAlchemy 2.0).

Stores per-document ingestion status, chunk metadata, and the full-text
(``tsvector``) index that becomes the BM25/lexical side of hybrid retrieval.

The connection target is pure config (``POSTGRES_DSN``) — the same code runs
against local Docker Postgres, Neon, Supabase, or RDS unchanged.
"""

from __future__ import annotations

from scholarrag.db.engine import get_db, get_engine, session_scope
from scholarrag.db.models import Base, Chunk, Document, IngestionStatus

__all__ = [
    "Base",
    "Chunk",
    "Document",
    "IngestionStatus",
    "get_db",
    "get_engine",
    "session_scope",
]
