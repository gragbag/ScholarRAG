"""Database repository tests.

These need a real Postgres (the generated ``tsvector`` column and enum types are
Postgres-specific — SQLite can't fake them). The shared ``db`` fixture (in
conftest.py) skips them automatically if Postgres isn't reachable.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from scholarrag.db import repository as repo
from scholarrag.db.models import Document, IngestionStatus
from scholarrag.db.repository import NewChunk


def _make_doc(
    session: Session,
    *,
    filename: str = "paper.pdf",
    content_hash: str | None = None,
    collection: str = "default",
    user_id: uuid.UUID | None = None,
) -> Document:
    return repo.create_document(
        session,
        filename=filename,
        content_hash=content_hash or uuid.uuid4().hex,
        content_type="pdf",
        corpus_profile="research_papers",
        collection=collection,
        user_id=user_id,
    )


def test_create_and_get_document(db: Session) -> None:
    doc = _make_doc(db, filename="vaswani2017.pdf")
    assert doc.id is not None
    assert doc.status is IngestionStatus.queued
    assert doc.num_chunks == 0

    fetched = repo.get_document(db, doc.id)
    assert fetched is not None
    assert fetched.filename == "vaswani2017.pdf"


def test_get_document_by_hash(db: Session) -> None:
    doc = _make_doc(db, content_hash="abc123")
    found = repo.get_document_by_hash(db, "abc123")
    assert found is not None
    assert found.id == doc.id
    assert repo.get_document_by_hash(db, "does-not-exist") is None


def test_duplicate_content_hash_rejected_within_owner_folder(db: Session) -> None:
    alice = repo.upsert_user(db, google_sub="dup-sub", email="dup@example.com")
    db.flush()
    _make_doc(db, content_hash="dup", collection="research", user_id=alice.id)
    # Same bytes, same owner, same folder → rejected by uq_documents_owner_collection_hash.
    with pytest.raises(IntegrityError):
        _make_doc(db, content_hash="dup", collection="research", user_id=alice.id)


def test_same_hash_allowed_across_folders(db: Session) -> None:
    alice = repo.upsert_user(db, google_sub="af-sub", email="af@example.com")
    db.flush()
    a = _make_doc(db, content_hash="shared", collection="research", user_id=alice.id)
    # Same bytes, different folder → allowed: a paper can live in two folders, and
    # a user's copy is separate from the public seed corpus's copy.
    b = _make_doc(db, content_hash="shared", collection="ml", user_id=alice.id)
    assert a.id != b.id


def test_set_document_status(db: Session) -> None:
    doc = _make_doc(db)
    repo.set_document_status(db, doc.id, IngestionStatus.failed, error="parse blew up")

    reloaded = repo.get_document(db, doc.id)
    assert reloaded is not None
    assert reloaded.status is IngestionStatus.failed
    assert reloaded.error == "parse blew up"


def test_add_chunks_and_generated_fts(db: Session) -> None:
    doc = _make_doc(db)
    written = repo.add_chunks(
        db,
        doc.id,
        [
            NewChunk(
                chunk_index=0,
                text="Neural networks learn representations.",
                vector_id="v0",
                char_count=38,
            ),
            NewChunk(
                chunk_index=1,
                text="Transformers rely on self-attention.",
                vector_id="v1",
                char_count=36,
            ),
        ],
    )
    assert written == 2
    assert repo.count_chunks(db, doc.id) == 2

    reloaded = repo.get_document(db, doc.id)
    assert reloaded is not None
    assert reloaded.num_chunks == 2

    # The generated tsvector column powers full-text (lexical/BM25) search.
    hits = db.execute(
        text("SELECT count(*) FROM chunks WHERE fts @@ to_tsquery('english', 'neural')")
    ).scalar_one()
    assert hits == 1


# @pytest.mark.skip(reason="Step 1 exercise — implement list_documents; see EXERCISES.md")
def test_list_documents(db: Session) -> None:
    _make_doc(db, filename="a.pdf")
    _make_doc(db, filename="b.pdf")
    _make_doc(db, filename="c.pdf")

    all_docs = repo.list_documents(db)
    assert len(all_docs) == 3

    limited = repo.list_documents(db, limit=2)
    assert len(limited) == 2
    # newest-first: created_at should be non-increasing
    assert all(limited[i].created_at >= limited[i + 1].created_at for i in range(len(limited) - 1))


def test_folder_summaries_groups_and_scopes(db: Session) -> None:
    alice = repo.upsert_user(db, google_sub="sub-alice", email="alice@example.com")
    bob = repo.upsert_user(db, google_sub="sub-bob", email="bob@example.com")
    db.flush()

    # Alice: 2 docs in "research", 1 in "ml". Bob: 1 in "research" (must NOT count).
    _make_doc(db, collection="research", user_id=alice.id)
    _make_doc(db, collection="research", user_id=alice.id)
    _make_doc(db, collection="ml", user_id=alice.id)
    _make_doc(db, collection="research", user_id=bob.id)

    summaries = repo.folder_summaries(db, user_id=alice.id)

    # Grouped + counted, scoped to Alice, ordered by folder name.
    assert summaries == [("ml", 1), ("research", 2)]
    assert dict(summaries) == {"ml": 1, "research": 2}


def test_create_folder_is_idempotent_per_user(db: Session) -> None:
    alice = repo.upsert_user(db, google_sub="sub-alice", email="alice@example.com")
    bob = repo.upsert_user(db, google_sub="sub-bob", email="bob@example.com")
    db.flush()

    f1 = repo.create_folder(db, user_id=alice.id, name="research")
    f2 = repo.create_folder(db, user_id=alice.id, name="research")  # same name → same row
    assert f1.id == f2.id

    # Different user, same name → a separate folder (uniqueness is per user).
    fb = repo.create_folder(db, user_id=bob.id, name="research")
    assert fb.id != f1.id

    assert repo.list_user_folders(db, alice.id) == ["research"]


def test_delete_document_is_owner_scoped(db: Session) -> None:
    alice = repo.upsert_user(db, google_sub="del-a", email="a@example.com")
    bob = repo.upsert_user(db, google_sub="del-b", email="b@example.com")
    db.flush()
    doc = _make_doc(db, collection="research", user_id=alice.id)

    # Bob can't delete Alice's document.
    assert repo.delete_document(db, doc.id, user_id=bob.id) is False
    assert repo.get_document(db, doc.id) is not None

    # Alice can — and it's really gone.
    assert repo.delete_document(db, doc.id, user_id=alice.id) is True
    assert repo.get_document(db, doc.id) is None
