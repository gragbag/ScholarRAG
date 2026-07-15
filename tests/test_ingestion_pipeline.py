"""End-to-end ingestion pipeline tests.

Real Postgres + FakeEmbedder + LocalVectorStore — no cloud, no model. The two
full-flow tests are skipped until you implement
``IngestionPipeline._build_records`` (Step 4 exercise); remove the
``@pytest.mark.skip`` on each to turn them on. ``test_ingest_unsupported_type``
passes now — it exercises the pipeline wiring without reaching the glue.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from scholarrag.corpus import get_corpus_profile
from scholarrag.db import repository as repo
from scholarrag.db.models import IngestionStatus
from scholarrag.embeddings import FakeEmbedder
from scholarrag.ingestion import IngestionPipeline, UnsupportedFileTypeError, content_hash
from scholarrag.vectorstore import LocalVectorStore

DIM = 32


def _doc_bytes() -> bytes:
    return (
        "Retrieval augmented generation grounds language models in retrieved "
        "documents. Hybrid retrieval combines dense and lexical search. " * 30
    ).encode()


def test_ingest_unsupported_type_raises(db: Session) -> None:
    # An unknown extension is rejected before any document row is created.
    pipe = IngestionPipeline(embedder=FakeEmbedder(dim=DIM), vector_store=LocalVectorStore(dim=DIM))
    profile = get_corpus_profile("generic_docs")

    with pytest.raises(UnsupportedFileTypeError):
        pipe.ingest(db, data=b"hello", filename="weird.xyz", profile=profile)

    assert repo.get_document_by_hash(db, content_hash(b"hello")) is None


def test_ingest_end_to_end(db: Session) -> None:
    embedder = FakeEmbedder(dim=DIM)
    store = LocalVectorStore(dim=DIM)
    pipe = IngestionPipeline(embedder=embedder, vector_store=store)
    profile = get_corpus_profile("generic_docs")

    result = pipe.ingest(db, data=_doc_bytes(), filename="paper.txt", profile=profile)

    assert result.skipped is False
    assert result.status is IngestionStatus.completed
    assert result.num_chunks > 1

    document = repo.get_document(db, result.document_id)
    assert document is not None
    assert document.status is IngestionStatus.completed
    assert repo.count_chunks(db, result.document_id) == result.num_chunks

    # One vector per chunk, carrying the chunk text as metadata.
    assert store.count() == result.num_chunks
    hits = store.query(embedder.embed_query("hybrid retrieval"), top_k=3)
    assert hits
    assert "text" in hits[0].metadata


def test_ingest_is_idempotent(db: Session) -> None:
    embedder = FakeEmbedder(dim=DIM)
    store = LocalVectorStore(dim=DIM)
    pipe = IngestionPipeline(embedder=embedder, vector_store=store)
    profile = get_corpus_profile("generic_docs")
    data = _doc_bytes()

    first = pipe.ingest(db, data=data, filename="paper.txt", profile=profile)
    second = pipe.ingest(db, data=data, filename="paper.txt", profile=profile)

    assert first.skipped is False
    assert second.skipped is True  # identical bytes → not re-ingested
    assert second.document_id == first.document_id
    assert repo.count_chunks(db, first.document_id) == first.num_chunks
