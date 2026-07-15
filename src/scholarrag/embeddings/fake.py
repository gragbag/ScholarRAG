"""Deterministic, dependency-free embedder for tests and CI.

No torch, no model download, no network — the test-suite counterpart to
``LocalVectorStore``. It's a *hashing bag-of-words*: each word bumps a bucket in
the vector, then the vector is L2-normalized. It has no real semantic
understanding, but texts that share words end up with higher cosine similarity,
which is enough to exercise the retrieval pipeline in later steps without ever
loading a model.
"""

from __future__ import annotations

import hashlib
import math
import re

from scholarrag.embeddings.base import Vector

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lower-case and split into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """A deterministic hashing embedder implementing the :class:`Embedder` protocol."""

    def __init__(self, *, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        """Turn ``text`` into a deterministic, L2-normalized vector of length ``dim``."""
        embeddings = [0.0] * self._dim
        for token in _tokenize(text):
            embeddings[int(hashlib.sha1(token.encode()).hexdigest(), 16) % self._dim] += 1.0

        magnitude = math.sqrt(sum(x * x for x in embeddings))
        if magnitude == 0.0:
            return embeddings

        return [x / magnitude for x in embeddings]
