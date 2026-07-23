"""Application configuration.

12-factor: every value comes from the environment (or a local, git-ignored
``.env``). A committed ``.env.example`` documents the full surface. No secrets
in the repo.

The one design rule worth calling out: nothing here forces a live external
dependency. If ``PINECONE_API_KEY`` is unset we fall back to the in-process
``LocalVectorStore`` so tests and CI stay free and hermetic (see
``scholarrag.vectorstore``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# "fake" = deterministic, dependency-free LLM client used in tests/CI.
LLMProvider = Literal["anthropic", "gemini", "openai", "ollama", "fake"]
# "fake" = deterministic, dependency-free embedder used in tests/CI.
EmbeddingProvider = Literal["local", "fake", "openai"]
VectorStoreKind = Literal["auto", "local", "pinecone"]
# "none" = fusion only (no rerank); "fake" = deterministic torch-free reranker.
RerankerProvider = Literal["cross_encoder", "fake", "none"]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables.

    Env var names are the upper-cased field names (e.g. ``LLM_PROVIDER``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- App -----------------------------------------------------------------
    app_name: str = "scholarrag"
    environment: Literal["local", "ci", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # -- LLM (provider-agnostic; Claude is the default) ----------------------
    # See docs/DESIGN.md for why Claude is the default and how the tiers map.
    llm_provider: LLMProvider = "anthropic"
    anthropic_api_key: str | None = None
    # Cheap tier: query rewriting / HyDE / classification.
    llm_model_cheap: str = "claude-haiku-4-5"
    # Strong tier: final grounded generation with citations.
    llm_model_strong: str = "claude-sonnet-5"
    llm_max_output_tokens: int = 1024

    # Query rewriting (Phase 2 Step 3): expand a query into N search variations
    # via the cheap tier. Off by default; Step 4 wires it into the pipeline.
    query_rewriting_enabled: bool = False
    num_query_variations: int = 3

    # Optional non-default providers (wired in Phase 2).
    gemini_api_key: str | None = None
    # Gemini's own model ids (the cheap/strong tiers map to these when
    # LLM_PROVIDER=gemini). Free-tier friendly defaults; run models.list to see
    # what your key can access.
    gemini_model_cheap: str = "gemini-3.1-flash-lite"
    gemini_model_strong: str = "gemini-3.5-flash"
    # Free-tier Gemini caps the strong model at ~5 req/min; the SDK doesn't retry
    # 429s, so GeminiLLM backs off and retries this many times before giving up.
    gemini_max_retries: int = 6
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # -- Embeddings (local sentence-transformers by default; free/on-device) -
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # -- Vector store --------------------------------------------------------
    # "auto" -> Pinecone when PINECONE_API_KEY is set, else LocalVectorStore.
    vector_store: VectorStoreKind = "auto"
    pinecone_api_key: str | None = None
    pinecone_index: str = "scholarrag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # -- Retrieval / reranking (Phase 2 Step 2) ------------------------------
    # Hybrid retrieval fuses dense + lexical with Reciprocal Rank Fusion, then
    # (optionally) a cross-encoder reranks the fused shortlist for precision.
    reranker_provider: RerankerProvider = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_candidate_k: int = 50  # stage-1 pool size fed to fusion/rerank
    rrf_k: int = 60  # RRF constant; larger => flatter weighting of top ranks
    retrieval_top_k: int = 5  # final chunks fed into the answer prompt (Step 4)

    # -- Observability (Phase 4 Step 1) --------------------------------------
    # Langfuse LLM tracing. All optional: tracing activates only when both keys
    # are set (create them in the Langfuse UI at http://localhost:3001 -> project
    # settings -> API keys) AND the `observability` extra is installed.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3001"
    # OpenTelemetry app tracing -> Jaeger (docker-compose, UI on :16686). Off
    # until the OTLP endpoint is set (and the `observability` extra installed).
    otel_exporter_endpoint: str | None = None  # e.g. http://localhost:4318
    otel_service_name: str = "scholarrag"

    # -- Guardrails (Phase 4 Step 3) -----------------------------------------
    # Per-client-IP fixed-window rate limit on /query (Redis). Off by default;
    # query length bounds are enforced by the request schema (3..2000 chars).
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 20

    # -- Answer cache (Phase 4 Step 2) ---------------------------------------
    # Exact + semantic caching of final answers in Redis. A hit skips retrieval
    # AND the LLM call. Off by default (needs Redis; flip on in .env).
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600  # invalidation-by-expiry; seed also clears
    semantic_cache_enabled: bool = True  # paraphrase matching within the cache
    semantic_cache_threshold: float = 0.93  # cosine floor; higher = fewer false hits

    # -- Evaluation (Phase 3) ------------------------------------------------
    eval_k: int = 5  # top-k cutoff for retrieval metrics
    eval_recall_threshold: float = 0.8  # CI/report floor for Recall@k on the golden set
    # Generation eval (Step 2): RAGAS is token-hungry, so bound the sample and
    # throttle concurrency for the free tier.
    eval_sample_size: int = 8  # examples per RAGAS run
    eval_max_workers: int = 2  # RAGAS judge concurrency (lower = gentler on rate limits)
    mlflow_tracking_uri: str | None = None  # e.g. http://localhost:5001; None = local ./mlruns

    # -- Infrastructure (Phase 1+) ------------------------------------------
    # Host ports are offset (5433/6380) to coexist with other local stacks; see
    # docker-compose.yml. Inside the compose network the api overrides these to
    # the standard container ports (postgres:5432 / redis:6379).
    postgres_dsn: str = "postgresql+psycopg://scholarrag:scholarrag@localhost:5433/scholarrag"
    redis_url: str = "redis://localhost:6380/0"

    # -- Corpus --------------------------------------------------------------
    # Selects a profile in scholarrag.corpus (domain is swappable).
    corpus_profile: str = "research_papers"

    @property
    def use_pinecone(self) -> bool:
        """Whether the resolved vector store should be Pinecone-backed."""
        if self.vector_store == "pinecone":
            return True
        if self.vector_store == "local":
            return False
        return self.pinecone_api_key is not None  # "auto"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
