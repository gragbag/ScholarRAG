"""Retrieval — dense (semantic) and lexical (keyword) search over the corpus.

Both implement the :class:`Retriever` protocol and return :class:`RetrievedChunk`s.
Step 2 fuses them (RRF) and reranks; here we just build the two engines.
"""

from __future__ import annotations

from scholarrag.retrieval.base import RetrievedChunk, Retriever
from scholarrag.retrieval.dense import DenseRetriever
from scholarrag.retrieval.lexical import LexicalRetriever

__all__ = [
    "DenseRetriever",
    "LexicalRetriever",
    "RetrievedChunk",
    "Retriever",
]
