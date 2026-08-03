"""Modal-hosted embeddings — runs BGE off-box so the backend carries no torch.

Phase C of the deploy rehaul: the embedding model runs as a serverless function
on Modal (see ``deploy/modal/embed_app.py``), and this client just POSTs text to
its web endpoint over ``httpx`` (a core dep). That's what keeps torch /
sentence-transformers OUT of the Cloud Run image — the backend never imports them.

The Modal service is deliberately dumb ("encode these texts, normalized"); the
query-prefix asymmetry (BGE wants an instruction prefix on queries, not passages)
stays here, exactly like :class:`LocalEmbedder` — so vectors match a locally
embedded corpus and the two are interchangeable.

``post_fn`` is a test seam: inject it to exercise the client without Modal, a URL,
or a network call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings
from scholarrag.embeddings.base import Vector

if TYPE_CHECKING:  # pragma: no cover
    import httpx

# post_fn(url, *, json=<payload>, headers=<dict>) -> the parsed JSON response dict.
PostFn = Callable[..., dict[str, Any]]


class ModalEmbedder:
    """Calls a Modal-hosted BGE endpoint. Implements the :class:`Embedder` protocol."""

    def __init__(
        self,
        settings: Settings,
        *,
        query_prefix: str = "",
        post_fn: PostFn | None = None,
    ) -> None:
        if settings.modal_embed_url is None and post_fn is None:
            raise ValueError("MODAL_EMBED_URL is required for ModalEmbedder")
        self._settings = settings
        self._query_prefix = query_prefix
        self._post_fn = post_fn
        self._client: httpx.Client | None = None

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    def _client_lazy(self) -> httpx.Client:
        if self._client is None:
            import httpx

            # Generous timeout: a cold Modal container (boot + model load) can take
            # tens of seconds on the first call after scale-to-zero; warm is <1s.
            self._client = httpx.Client(timeout=self._settings.modal_embed_timeout)
        return self._client

    def _embed(self, texts: list[str]) -> list[Vector]:
        """POST texts to the Modal endpoint and return their embeddings."""
        payload = {"texts": texts}
        headers: dict[str, str] = {}
        if self._settings.modal_embed_token is not None:
            headers["Authorization"] = f"Bearer {self._settings.modal_embed_token}"

        if self._post_fn is not None:
            data = self._post_fn(self._settings.modal_embed_url, json=payload, headers=headers)
        else:
            url = self._settings.modal_embed_url
            assert url is not None  # __init__ guarantees this on the real path
            response = self._client_lazy().post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        embeddings: list[Vector] = data["embeddings"]
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> Vector:
        # Prepend BGE's query instruction here (the service stays symmetric).
        return self._embed([self._query_prefix + text])[0]
