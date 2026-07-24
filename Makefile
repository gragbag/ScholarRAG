# ScholarRAG — reproducible developer entry points.
# Everything runs through `uv` so the environment is pinned and hermetic.

.DEFAULT_GOAL := help
.PHONY: help install lint fmt type test check run seed eval eval-gen eval-rag eval-agentic up down logs clean

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
	# --all-extras keeps the embeddings (BGE) and llm (Claude/Gemini) SDKs
	# installed; a bare `uv run` re-syncs to core deps and uninstalls them.
	uv run --all-extras uvicorn --factory scholarrag.api.main:create_app --reload --host 0.0.0.0 --port 8001

ui: ## Launch the Streamlit chat UI (needs the API running — `make run` — in another terminal)
	uv run --all-extras streamlit run src/scholarrag/ui/app.py

seed: ## Ingest the sample corpus (synchronous; needs Postgres + the embeddings extra)
	uv run --all-extras python -m scholarrag.scripts.seed

eval: ## Run retrieval eval over the golden set (needs Postgres + a seeded corpus)
	uv run --all-extras python -m scholarrag.scripts.eval

eval-gen: ## Generate a synthetic eval set with the LLM (offline; costs a few tokens)
	uv run --all-extras python -m scholarrag.scripts.gen_eval

eval-rag: ## Generation eval with RAGAS + MLflow (needs seeded corpus; spends free-tier tokens)
	uv run --all-extras python -m scholarrag.scripts.eval_rag

eval-agentic: ## Agentic vs single-shot on the hard set (both pipelines; slow, rate-limited)
	uv run --all-extras python -m scholarrag.scripts.eval_agentic

up: ## Boot the full stack (API, Postgres, Redis, Langfuse, MLflow)
	docker compose up -d --build

down: ## Stop the stack
	docker compose down

logs: ## Tail stack logs
	docker compose logs -f

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
