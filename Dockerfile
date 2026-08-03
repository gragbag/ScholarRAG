# Multi-stage build for the ScholarRAG API.
# Stage 1 resolves and installs deps with uv into a venv; stage 2 is a slim
# runtime that copies only that venv and the source.

FROM python:3.12-slim AS builder

# uv: fast, reproducible installs from the lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# The extras the running services actually need. ONE image, THREE roles: the same
# venv runs uvicorn (API), the Celery worker, or Streamlit (UI), so it must carry
# the union of their deps — embeddings + llm + langchain + agentic + ui +
# observability. `eval` (ragas/mlflow/datasets) is dev/CI-only and deliberately
# left out to keep the image lean. Without these, the image ModuleNotFounds at
# query time (the "uv strips extras" gotcha, baked in).
ARG EXTRAS="--extra llm --extra langchain --extra agentic --extra observability --extra auth --extra html --extra cloudtasks"

WORKDIR /app

# Install dependencies first (cached layer), without the project itself.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev ${EXTRAS}

# Now install the project.
COPY src ./src
COPY README.md ./
# Alembic migrations, so the image can create its own schema (`alembic upgrade
# head`) — a production image should be able to migrate the DB it deploys against.
COPY alembic.ini ./
COPY alembic ./alembic
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${EXTRAS}


FROM python:3.12-slim AS runtime

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:800/health').status_code==200 else 1)"

CMD ["sh", "-c", "uvicorn --factory scholarrag.api.main:create_app --host 0.0.0.0 --port ${PORT:-8000}"]
