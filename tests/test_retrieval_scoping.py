"""Folder-scoped retrieval tests (Phase 8, Step 1).

The first test passes now (the `collection` plumbing). The two skipped ones are
the exercise targets — dense + lexical scoping — in retrieval/dense.py and
retrieval/lexical.py.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from scholarrag.db import repository as repo
from scholarrag.db.repository import NewChunk
from scholarrag.embeddings.fake import FakeEmbedder
from scholarrag.retrieval.dense import DenseRetriever
from scholarrag.retrieval.lexical import LexicalRetriever
from scholarrag.vectorstore.base import VectorRecord
from scholarrag.vectorstore.local import LocalVectorStore

DIM = 8


def _doc(
    session: Session,
    *,
    filename: str,
    digest: str,
    collection: str = "default",
    user_id: uuid.UUID | None = None,
) -> uuid.UUID:
    doc = repo.create_document(
        session,
        filename=filename,
        content_hash=digest,
        content_type="txt",
        corpus_profile="research_papers",
        collection=collection,
        user_id=user_id,
    )
    session.flush()
    return doc.id


def _owned_record(
    embedder: FakeEmbedder, doc_id: uuid.UUID, filename: str, owner: str
) -> VectorRecord:
    (vec,) = embedder.embed_documents([filename])
    return VectorRecord(
        id=f"{doc_id}:0",
        values=vec,
        metadata={
            "text": filename,
            "document_id": str(doc_id),
            "chunk_index": 0,
            "filename": filename,
            "collection": "default",
            "owner": owner,
        },
    )


# ── passes now: the collection plumbing ──────────────────────────────────────
def test_documents_carry_collection_and_list(db: Session) -> None:
    _doc(db, filename="a", digest="c1", collection="alpha")
    _doc(db, filename="b", digest="c2", collection="beta")
    assert set(repo.list_collections(db)) == {"alpha", "beta"}


def test_dense_scoping(db: Session) -> None:
    embedder = FakeEmbedder(dim=DIM)
    store = LocalVectorStore(dim=DIM)
    da, db_id = uuid.uuid4(), uuid.uuid4()
    va, vb = embedder.embed_documents(["alpha", "beta"])
    store.upsert(
        [
            VectorRecord(
                id=f"{da}:0",
                values=va,
                metadata={
                    "text": "alpha",
                    "document_id": str(da),
                    "chunk_index": 0,
                    "filename": "a.txt",
                    "collection": "A",
                },
            ),
            VectorRecord(
                id=f"{db_id}:0",
                values=vb,
                metadata={
                    "text": "beta",
                    "document_id": str(db_id),
                    "chunk_index": 0,
                    "filename": "b.txt",
                    "collection": "B",
                },
            ),
        ]
    )
    retriever = DenseRetriever(embedder=embedder, vector_store=store)

    scoped = retriever.retrieve(db, "anything", top_k=10, collection="A")
    assert {h.filename for h in scoped} == {"a.txt"}  # folder A only

    unscoped = retriever.retrieve(db, "anything", top_k=10)
    assert {h.filename for h in unscoped} == {"a.txt", "b.txt"}  # None = all folders


# ── Exercise 2 — lexical folder scoping (retrieval/lexical.py) ────────────────
def test_lexical_scoping(db: Session) -> None:
    doc_a = _doc(db, filename="a.txt", digest="h-a", collection="A")
    doc_b = _doc(db, filename="b.txt", digest="h-b", collection="B")
    repo.add_chunks(
        db,
        doc_a,
        [NewChunk(chunk_index=0, text="quantum computing", vector_id=f"{doc_a}:0", char_count=17)],
    )
    repo.add_chunks(
        db,
        doc_b,
        [NewChunk(chunk_index=0, text="quantum mechanics", vector_id=f"{doc_b}:0", char_count=17)],
    )
    db.flush()
    retriever = LexicalRetriever()

    scoped = retriever.retrieve(db, "quantum", top_k=10, collection="A")
    assert {h.filename for h in scoped} == {"a.txt"}  # folder A only

    unscoped = retriever.retrieve(db, "quantum", top_k=10)
    assert {h.filename for h in unscoped} == {"a.txt", "b.txt"}  # None = all folders


# ── Exercise B — per-user (owner) scoping ────────────────────────────────────
def test_dense_owner_scoping(db: Session) -> None:
    embedder = FakeEmbedder(dim=DIM)
    store = LocalVectorStore(dim=DIM)
    ua, ub = uuid.uuid4(), uuid.uuid4()
    da, db_id = uuid.uuid4(), uuid.uuid4()
    store.upsert(
        [
            _owned_record(embedder, da, "a.txt", str(ua)),
            _owned_record(embedder, db_id, "b.txt", str(ub)),
        ]
    )
    retriever = DenseRetriever(embedder=embedder, vector_store=store)

    scoped = retriever.retrieve(db, "anything", top_k=10, user_id=ua)
    assert {h.filename for h in scoped} == {"a.txt"}  # only user A's docs

    anon = retriever.retrieve(db, "anything", top_k=10)  # None = anonymous sees all
    assert {h.filename for h in anon} == {"a.txt", "b.txt"}


def test_lexical_owner_scoping(db: Session) -> None:
    ua = repo.upsert_user(db, google_sub="ua", email="a@x.com")
    ub = repo.upsert_user(db, google_sub="ub", email="b@x.com")
    db.flush()
    doc_a = _doc(db, filename="a.txt", digest="o-a", user_id=ua.id)
    doc_b = _doc(db, filename="b.txt", digest="o-b", user_id=ub.id)
    repo.add_chunks(
        db,
        doc_a,
        [NewChunk(chunk_index=0, text="quantum computing", vector_id=f"{doc_a}:0", char_count=17)],
    )
    repo.add_chunks(
        db,
        doc_b,
        [NewChunk(chunk_index=0, text="quantum mechanics", vector_id=f"{doc_b}:0", char_count=17)],
    )
    db.flush()
    retriever = LexicalRetriever()

    scoped = retriever.retrieve(db, "quantum", top_k=10, user_id=ua.id)
    assert {h.filename for h in scoped} == {"a.txt"}  # only user A's docs

    anon = retriever.retrieve(db, "quantum", top_k=10)  # anonymous sees all
    assert {h.filename for h in anon} == {"a.txt", "b.txt"}
