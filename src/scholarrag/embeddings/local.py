"""Local embeddings via sentence-transformers (the production default).

Runs a BGE model on-device — free, private, no API calls. The model is **lazy
loaded**: constructing a :class:`LocalEmbedder` is cheap, and the ~130 MB model
is only downloaded/loaded the first time you actually embed something. That's
what keeps imports fast and lets tests construct one without pulling in torch.

Requires the ``embeddings`` extra (``uv sync --extra embeddings``); tests use
:class:`~scholarrag.embeddings.fake.FakeEmbedder` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scholarrag.embeddings.base import Vector

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

# BGE retrieval convention: prefix the QUERY with an instruction; leave passages
# bare. This asymmetry measurably improves retrieval quality.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Sentence-transformers implementation of the :class:`Embedder` protocol."""

    def __init__(self, model_name: str, *, dim: int, query_prefix: str = "") -> None:
        self._model_name = model_name
        self._dim = dim
        self._query_prefix = query_prefix
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        """Load (and cache) the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        model = self._load()
        # normalize_embeddings=True -> unit vectors, so cosine == dot product.
        matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in matrix]

    def embed_query(self, text: str) -> Vector:
        model = self._load()
        vector = model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(x) for x in vector]
