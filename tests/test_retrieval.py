"""Retriever tests.

``test_retrieved_chunk`` and the protocol-conformance test pass now. The two
behaviour tests are the Step 1 exercises — remove each ``@pytest.mark.skip`` once
you've implemented the corresponding ``retrieve``.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from scholarrag.db import repository as repo
from scholarrag.db.repository import NewChunk
from scholarrag.embeddings import FakeEmbedder
from scholarrag.retrieval import (
    DenseRetriever,
    LexicalRetriever,
    RetrievedChunk,
    Retriever,
)
from scholarrag.vectorstore import LocalVectorStore, VectorRecord

DIM = 32


def test_retrieved_chunk_dataclass() -> None:
    c = RetrievedChunk(
        id="doc:0", document_id=uuid.uuid4(), chunk_index=0, text="hi", filename="a.txt", score=0.5
    )
    assert (c.id, c.chunk_index, c.score) == ("doc:0", 0, 0.5)


def test_retrievers_conform_to_protocol() -> None:
    dense = DenseRetriever(embedder=FakeEmbedder(dim=DIM), vector_store=LocalVectorStore(dim=DIM))
    assert isinstance(dense, Retriever)
    assert isinstance(LexicalRetriever(), Retriever)


# ── Exercise A — DenseRetriever.retrieve (hermetic; no Postgres) ─────────────
def test_dense_retriever_ranks_by_meaning() -> None:
    emb = FakeEmbedder(dim=DIM)
    store = LocalVectorStore(dim=DIM)
    doc_id = uuid.uuid4()
    texts = [
        "neural networks learn representations with attention",
        "a shortbread recipe with butter and sugar",
    ]
    vectors = emb.embed_documents(texts)
    store.upsert(
        [
            VectorRecord(
                id=f"{doc_id}:{i}",
                values=vectors[i],
                metadata={
                    "text": texts[i],
                    "document_id": str(doc_id),
                    "chunk_index": i,
                    "filename": "a.txt",
                },
            )
            for i in range(2)
        ]
    )
    retriever = DenseRetriever(embedder=emb, vector_store=store)

    results = retriever.retrieve(Session(), "how do neural networks work", top_k=2)
    assert results[0].id == f"{doc_id}:0"  # the neural chunk, not the recipe
    assert results[0].text == texts[0]
    assert results[0].document_id == doc_id


# ── Exercise B — LexicalRetriever.retrieve (needs Postgres) ──────────────────
def test_lexical_retriever_finds_keyword(db: Session) -> None:
    doc = repo.create_document(
        db,
        filename="paper.txt",
        content_hash="lex-hash-1",
        content_type="txt",
        corpus_profile="generic_docs",
    )
    repo.add_chunks(
        db,
        doc.id,
        [
            NewChunk(
                chunk_index=0,
                text="Neural networks learn representations via gradient descent.",
                vector_id=f"{doc.id}:0",
                char_count=59,
            ),
            NewChunk(
                chunk_index=1,
                text="A shortbread recipe uses butter, sugar, and flour.",
                vector_id=f"{doc.id}:1",
                char_count=50,
            ),
        ],
    )

    results = LexicalRetriever().retrieve(db, "neural networks", top_k=5)
    assert results
    assert results[0].id == f"{doc.id}:0"
    assert "neural" in results[0].text.lower()
