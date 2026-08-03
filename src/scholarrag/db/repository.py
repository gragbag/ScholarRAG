"""Repository — typed, testable functions for reading/writing the tables.

Every function takes an explicit ``Session`` and only ``flush``es (see
``engine.py`` for why). Keeping SQL access here — rather than sprinkled through
routes and workers — means one place to reason about queries, and one place to
test them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from scholarrag.db.models import Chunk, Document, Folder, IngestionStatus, User


@dataclass(frozen=True, slots=True)
class NewChunk:
    """Input for :func:`add_chunks` — the chunker (Step 3) produces these."""

    chunk_index: int
    text: str
    vector_id: str
    char_count: int


def create_document(
    session: Session,
    *,
    filename: str,
    content_hash: str,
    content_type: str,
    corpus_profile: str,
    collection: str = "default",
    user_id: uuid.UUID | None = None,
) -> Document:
    """Insert a new document row in the ``queued`` state and return it."""
    document = Document(
        filename=filename,
        content_hash=content_hash,
        content_type=content_type,
        corpus_profile=corpus_profile,
        collection=collection,
        user_id=user_id,
        status=IngestionStatus.queued,
    )
    session.add(document)
    session.flush()  # assigns document.id without committing
    return document


def list_collections(session: Session, user_id: uuid.UUID | None = None) -> list[str]:
    """Return the distinct folder names (scoped to ``user_id`` when given)."""
    stmt = select(Document.collection).distinct().order_by(Document.collection)
    if user_id is not None:
        stmt = stmt.where(Document.user_id == user_id)
    return list(session.scalars(stmt).all())


def folder_summaries(session: Session, user_id: uuid.UUID | None = None) -> list[tuple[str, int]]:
    "Return each folder with its document count, as ``(name, count)`` pairs."

    stmt = (
        select(Document.collection, func.count(Document.id))
        .group_by(Document.collection)
        .order_by(Document.collection)
    )

    if user_id is not None:
        stmt = stmt.where(Document.user_id == user_id)

    rows = session.execute(stmt).all()
    return [(name, count) for name, count in rows]


def list_user_folders(session: Session, user_id: uuid.UUID) -> list[str]:
    """Names of folders a user has explicitly created (rows in the folders table).

    Distinct from ``list_collections``, which only surfaces folders that already
    contain a document. The route unions the two so empty folders show up too.
    """
    stmt = select(Folder.name).where(Folder.user_id == user_id).order_by(Folder.name)
    return list(session.scalars(stmt).all())


def create_folder(session: Session, *, user_id: uuid.UUID, name: str) -> Folder:
    "Register a folder for a user, returning it. Idempotent: the same"
    folder_search = select(Folder).where(Folder.user_id == user_id, Folder.name == name)
    found = session.scalars(folder_search).first()
    if found:
        return found

    new_folder = Folder(user_id=user_id, name=name)
    session.add(new_folder)
    session.flush()

    return new_folder


def upsert_user(session: Session, *, google_sub: str, email: str, name: str | None = None) -> User:
    """Return the user for ``google_sub``, creating them on first sign-in."""
    user = session.scalars(select(User).where(User.google_sub == google_sub)).first()
    if user is None:
        user = User(google_sub=google_sub, email=email, name=name)
        session.add(user)
        session.flush()
    else:
        user.email = email  # keep email fresh; google_sub is the stable key
        if name:
            user.name = name
    return user


def get_user(session: Session, user_id: uuid.UUID) -> User | None:
    """Load a user by primary key."""
    return session.get(User, user_id)


def get_document(session: Session, document_id: uuid.UUID) -> Document | None:
    """Fetch a document by primary key, or None."""
    return session.get(Document, document_id)


def get_document_by_hash(
    session: Session,
    content_hash: str,
    *,
    user_id: uuid.UUID | None = None,
    collection: str = "default",
) -> Document | None:
    """Idempotency lookup, scoped to one owner + folder: have these exact bytes
    already been added by this user to this collection?

    Scoping matters now that the same paper can live in different folders / belong
    to different users — and a user's own copy is separate from the public seed
    corpus (``user_id`` NULL). The defaults reproduce the old seed behaviour.
    """
    stmt = select(Document).where(
        Document.content_hash == content_hash,
        Document.collection == collection,
    )
    if user_id is None:
        stmt = stmt.where(Document.user_id.is_(None))
    else:
        stmt = stmt.where(Document.user_id == user_id)
    return session.scalars(stmt).one_or_none()


def set_document_status(
    session: Session,
    document_id: uuid.UUID,
    status: IngestionStatus,
    *,
    error: str | None = None,
) -> None:
    """Update a document's ingestion status (and optional error message)."""
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError(f"document {document_id} not found")
    document.status = status
    document.error = error
    session.flush()


def add_chunks(
    session: Session,
    document_id: uuid.UUID,
    new_chunks: list[NewChunk],
) -> int:
    """Bulk-insert chunks for a document and bump its ``num_chunks``.

    The ``fts`` tsvector is populated by Postgres automatically (generated
    column), so we never set it here.
    """
    rows = [
        Chunk(
            document_id=document_id,
            chunk_index=c.chunk_index,
            text=c.text,
            vector_id=c.vector_id,
            char_count=c.char_count,
        )
        for c in new_chunks
    ]
    session.add_all(rows)
    session.flush()

    document = session.get(Document, document_id)
    if document is not None:
        document.num_chunks = (document.num_chunks or 0) + len(rows)
        session.flush()
    return len(rows)


def count_chunks(session: Session, document_id: uuid.UUID) -> int:
    """Number of chunks stored for a document."""
    stmt = select(Chunk).where(Chunk.document_id == document_id)
    return len(session.scalars(stmt).all())


def list_documents(
    session: Session,
    *,
    collection: str | None = None,
    user_id: uuid.UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Document]:
    """Return documents newest-first, paginated by ``limit`` / ``offset``.

    Optionally scoped to a ``collection`` (folder) and/or ``user_id`` (owner) —
    the folder view in the extension passes both so a user sees only their own
    pages in the selected folder.
    """
    stmt = select(Document).order_by(Document.created_at.desc())
    if collection is not None:
        stmt = stmt.where(Document.collection == collection)
    if user_id is not None:
        stmt = stmt.where(Document.user_id == user_id)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.scalars(stmt).all())


def document_vector_ids(session: Session, document_id: uuid.UUID) -> list[str]:
    """The vector-store ids of a document's chunks — needed to purge them from the
    vector store when the document is deleted (the DB rows cascade; vectors don't)."""
    stmt = select(Chunk.vector_id).where(Chunk.document_id == document_id)
    return list(session.scalars(stmt).all())


def rename_document(
    session: Session, document_id: uuid.UUID, *, user_id: uuid.UUID | None, title: str
) -> Document | None:
    """Rename a document (its display ``filename``), scoped to the owner. Returns the
    updated document, or None if it doesn't exist or isn't the caller's."""
    document = session.get(Document, document_id)
    if document is None or document.user_id != user_id:
        return None
    document.filename = title
    session.flush()
    return document


def delete_document(session: Session, document_id: uuid.UUID, *, user_id: uuid.UUID | None) -> bool:
    "Delete a document (its chunks cascade in the DB), scoped to the owner. Returns"

    document = session.get(Document, document_id)
    if document is None or document.user_id != user_id:
        return False

    session.delete(document)
    session.flush()
    return True
