"""Retrieval-eval CI gate — hermetic (FakeEmbedder + LocalVectorStore, no Postgres).

Builds a tiny in-memory corpus whose questions share keywords with the right doc,
runs ``evaluate_retrieval``, and asserts Recall@k clears a floor. This catches
metric/retriever *logic* regressions deterministically. Real-corpus numbers come
from a local ``make eval`` against BGE + Pinecone.

Skipped until the metric exercises (A/B/C) are implemented, since the runner
computes all four metrics.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from scholarrag.embeddings import FakeEmbedder
from scholarrag.eval import EvalExample, evaluate_retrieval
from scholarrag.retrieval import DenseRetriever
from scholarrag.vectorstore import LocalVectorStore, VectorRecord

DIM = 64

# A id/text/filename corpus the FakeEmbedder can separate by shared words.
CORPUS = [
    ("rag.md", "retrieval augmented generation grounds answers and reduces hallucination"),
    ("embeddings.md", "an embedding model maps text to a vector for semantic similarity"),
    ("transformers.txt", "self attention lets each token attend to every other token"),
]

EXAMPLES = [
    EvalExample("how does retrieval augmented generation reduce hallucination", ["rag.md"]),
    EvalExample("what does an embedding model map text to", ["embeddings.md"]),
    EvalExample("what does self attention let each token do", ["transformers.txt"]),
]


def _dense_retriever() -> DenseRetriever:
    embedder = FakeEmbedder(dim=DIM)
    store = LocalVectorStore(dim=DIM)
    vectors = embedder.embed_documents([text for _, text in CORPUS])
    store.upsert(
        [
            VectorRecord(
                id=f"{uuid.uuid4()}:0",
                values=vectors[i],
                metadata={
                    "text": text,
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                    "filename": filename,
                },
            )
            for i, (filename, text) in enumerate(CORPUS)
        ]
    )
    return DenseRetriever(embedder=embedder, vector_store=store)


def test_retrieval_eval_meets_recall_floor() -> None:
    report = evaluate_retrieval(_dense_retriever(), Session(), EXAMPLES, k=3)
    assert report.num_queries == 3
    assert report.recall_at_k >= 0.8  # keyword-overlapping questions find the right doc
    assert report.mrr >= 0.8
