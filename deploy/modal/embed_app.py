"""ScholarRAG embedding service on Modal.

Runs BGE (sentence-transformers) as a serverless function so the main backend
carries no torch. This file is deployed to Modal's infrastructure — it is NOT
imported by the app (the backend's ModalEmbedder calls it over HTTP).

Setup (once you have a free Modal account + `modal setup`):

    # a shared secret the endpoint checks; use the same value as MODAL_EMBED_TOKEN
    modal secret create scholarrag-embed EMBED_TOKEN=<a-long-random-string>
    make modal-deploy            # or: modal deploy deploy/modal/embed_app.py

Modal prints the web-endpoint URL — put it in .env as MODAL_EMBED_URL, set
MODAL_EMBED_TOKEN to the secret, and EMBEDDING_PROVIDER=modal.

The model is baked into the image at build time, so cold starts don't re-download
it. It scales to zero; `scaledown_window` keeps a warm container briefly so
back-to-back queries don't each eat a cold start. Uncomment `gpu=` for GPU.
"""

# mypy: ignore-errors  # a deploy script against the untyped Modal SDK; not in `mypy src tests`
from __future__ import annotations

import os

import modal
from fastapi import Header, HTTPException  # core dep -> importable locally for `modal deploy`

MODEL_NAME = "BAAI/bge-small-en-v1.5"

app = modal.App("scholarrag-embeddings")


def _download_model() -> None:
    """Run at image-build time so the weights are baked into the image layer."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers==3.3.1",
        "torch==2.5.1",
        "fastapi[standard]",
        # CPU wheel keeps the image small (no CUDA); drop this for GPU inference.
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .run_function(_download_model)
)


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("scholarrag-embed")],
    scaledown_window=300,  # keep a warm container 5 min after the last call
    # BGE encoding is CPU-bound. Modal's default CPU is a fraction of a core, so
    # long (512-token) chunks crawled (~400ms each -> ~60s for a big doc). Request
    # real cores; torch parallelizes the batch across them.
    cpu=8.0,
    # gpu="T4",   # alternative: sub-second even for big docs (and often cheaper,
    #             # since it finishes so fast) — swap `cpu` for this to go GPU.
)
class EmbeddingModel:
    @modal.enter()
    def load(self) -> None:
        """Load the model once per container (not per request)."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)

    # `fastapi_endpoint` is the current name; older Modal calls it `web_endpoint`.
    @modal.fastapi_endpoint(method="POST")
    def embed(
        self,
        data: dict[str, list[str]],
        authorization: str = Header(default="", alias="Authorization"),
    ) -> dict[str, list[list[float]]]:
        expected = os.environ["EMBED_TOKEN"]
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="unauthorized")

        # normalize_embeddings=True -> unit vectors, matching LocalEmbedder exactly.
        vectors = self.model.encode(data["texts"], normalize_embeddings=True).tolist()
        return {"embeddings": vectors}
