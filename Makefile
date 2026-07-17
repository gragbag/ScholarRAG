# ScholarRAG — reproducible developer entry points.
# Everything runs through `uv` so the environment is pinned and hermetic.

.DEFAULT_GOAL := help
.PHONY: help install lint fmt type test check run up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install all deps (uv sync)
	uv sync --all-extras

fmt: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (ruff) + type-check (mypy)
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests

type: ## Type-check only (mypy)
	uv run mypy src tests

test: ## Run the test suite (LocalVectorStore; no cloud deps)
	uv run pytest

check: lint test ## Everything CI runs

run: ## Run the API locally with autoreload (port 8001 to avoid conflicts)
	uv run uvicorn scholarrag.api.main:app --reload --host 0.0.0.0 --port 8001

seed: ## Ingest the sample corpus (synchronous; needs Postgres + the embeddings extra)
	uv run python -m scholarrag.scripts.seed

up: ## Boot the full stack (API, Postgres, Redis, Langfuse, MLflow)
	docker compose up -d --build

down: ## Stop the stack
	docker compose down

logs: ## Tail stack logs
	docker compose logs -f

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
