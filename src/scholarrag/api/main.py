"""FastAPI application entry point.

Phase 0 exposes health/readiness and a small ``/info`` endpoint that reports the
resolved configuration (without secrets). Ingestion, retrieval, and generation
routes are added in later phases.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from scholarrag import __version__
from scholarrag.api.middleware import correlation_id_middleware
from scholarrag.api.routes import documents, query
from scholarrag.config import Settings, get_settings
from scholarrag.corpus import available_profiles, get_corpus_profile
from scholarrag.guardrails import build_rate_limiter
from scholarrag.logging import configure_logging, get_logger
from scholarrag.observability import configure_observability, configure_otel, is_otel_enabled
from scholarrag.observability import flush as flush_observability
from scholarrag.observability import is_enabled as observability_enabled
from scholarrag.vectorstore import build_vector_store

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str


class CorpusProfileResponse(BaseModel):
    name: str
    description: str
    chunk_size: int
    chunk_overlap: int
    file_types: list[str]


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]


class InfoResponse(BaseModel):
    app_name: str
    environment: str
    version: str
    llm_provider: str
    llm_model_cheap: str
    llm_model_strong: str
    embedding_provider: str
    embedding_model: str
    vector_store: str
    corpus_profile: str
    corpus_profiles_available: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Use the settings injected into create_app (app.state), NOT get_settings():
    # tests pass hermetic settings, and re-reading the developer's .env here
    # would flip on real telemetry mid-test-suite. (Observability itself is
    # configured in create_app — FastAPI instrumentation must attach BEFORE the
    # middleware stack is built; by lifespan time it's too late and the OTel
    # middleware would silently never run.)
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger.info(
        "startup",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        vector_store="pinecone" if settings.use_pinecone else "local",
        corpus_profile=settings.corpus_profile,
        observability=observability_enabled(),
        otel=is_otel_enabled(),
    )
    yield
    flush_observability()  # send any buffered traces before the process exits
    logger.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory — used by uvicorn and by the test suite."""
    settings = settings or get_settings()
    app = FastAPI(
        title="ScholarRAG",
        version=__version__,
        summary="Retrieval-Augmented Generation platform",
        lifespan=lifespan,
    )
    app.state.settings = settings  # the lifespan (and tests) read this back
    # Built from the injected settings (not env), so tests stay hermetic.
    app.state.rate_limiter = build_rate_limiter(settings)  # None when disabled
    configure_observability(settings)  # no-op unless Langfuse keys are set
    configure_otel(settings, app)  # must run pre-start so the middleware attaches
    app.middleware("http")(correlation_id_middleware)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness: the process is up and serving."""
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
    async def ready() -> ReadinessResponse:
        """Readiness: dependencies are reachable enough to serve traffic.

        In Phase 0 the only hard dependency is a constructable vector store; the
        local fallback makes this always true without external services.
        """
        checks: dict[str, bool] = {}
        try:
            build_vector_store(settings)
            checks["vector_store"] = True
        except Exception:  # pragma: no cover - defensive
            logger.exception("readiness_vector_store_failed")
            checks["vector_store"] = False
        return ReadinessResponse(ready=all(checks.values()), checks=checks)

    @app.get("/info", response_model=InfoResponse, tags=["ops"])
    async def info() -> InfoResponse:
        """Resolved, non-secret configuration — handy for demos and debugging."""
        profile = get_corpus_profile(settings.corpus_profile)
        return InfoResponse(
            app_name=settings.app_name,
            environment=settings.environment,
            version=__version__,
            llm_provider=settings.llm_provider,
            llm_model_cheap=settings.llm_model_cheap,
            llm_model_strong=settings.llm_model_strong,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            vector_store="pinecone" if settings.use_pinecone else "local",
            corpus_profile=profile.name,
            corpus_profiles_available=list(available_profiles()),
        )

    @app.get("/corpus/{name}", response_model=CorpusProfileResponse, tags=["corpus"])
    async def get_corpus(name: str) -> CorpusProfileResponse:
        try:
            profile = get_corpus_profile(name)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown corpus profile: {name}",
            ) from None

        return CorpusProfileResponse(
            name=profile.name,
            description=profile.description,
            chunk_size=profile.chunk_size,
            chunk_overlap=profile.chunk_overlap,
            file_types=list(profile.file_types),
        )

    app.include_router(documents.router)
    app.include_router(query.router)
    return app


# No module-level ``app = create_app()``: uvicorn runs the factory directly
# (``uvicorn --factory scholarrag.api.main:create_app``), so importing this
# module has no side effects — tests never trigger real telemetry by accident.
