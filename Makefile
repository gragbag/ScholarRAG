# ScholarRAG — reproducible developer entry points.
# Everything runs through `uv` so the environment is pinned and hermetic.

.DEFAULT_GOAL := help
.PHONY: help migrate install lint fmt type test check run seed eval eval-gen eval-rag eval-agentic up down logs ollama-up ollama-down clean ui cluster-up cluster-down k8s-image k8s-secret k8s-deploy k8s-status k8s-seed helm-lint helm-template helm-deploy helm-uninstall tf-init tf-validate tf-plan tf-apply tf-destroy

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

migrate: ## Apply DB schema migrations (alembic upgrade head)
	uv run alembic upgrade head

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

# ── Ollama (local, free LLM — set LLM_PROVIDER=ollama) ──────────────────────
# Pull once into the `ollama` volume; models persist across restarts. On a slow
# CPU, drop the strong model to a smaller tag (OLLAMA_MODEL_STRONG=llama3.2:3b).
OLLAMA_MODELS = llama3.2:3b llama3.1:8b
ollama-up: ## Start the Ollama container + pull the models (idempotent)
	docker compose --profile ollama up -d --wait ollama
	@for m in $(OLLAMA_MODELS); do echo "→ pulling $$m"; docker compose exec -T ollama ollama pull $$m; done

ollama-down: ## Stop the Ollama container (models stay in the volume)
	docker compose --profile ollama stop ollama

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ── Kubernetes / kind (Phase 7) ─────────────────────────────────────────────
cluster-up: ## Create the local kind cluster + NGINX ingress controller
	kind create cluster --config deploy/kind/cluster.yaml
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	kubectl -n ingress-nginx wait --for=condition=Ready pod \
		-l app.kubernetes.io/component=controller --timeout=180s

k8s-image: ## Build the app image and load it into the kind cluster
	docker compose build api
	kind load docker-image scholarrag-api:latest --name scholarrag

k8s-secret: ## Create/update the app Secret (Gemini + Pinecone keys) from .env
	kubectl -n scholarrag create secret generic scholarrag-secrets \
		--from-literal=GEMINI_API_KEY="$$(grep -E '^GEMINI_API_KEY=' .env | cut -d= -f2-)" \
		--from-literal=PINECONE_API_KEY="$$(grep -E '^PINECONE_API_KEY=' .env | cut -d= -f2-)" \
		--dry-run=client -o yaml | kubectl apply -f -

k8s-deploy: ## Apply namespace, secret, then all manifests; wait for the API
	kubectl apply -f deploy/k8s/00-namespace.yaml
	$(MAKE) k8s-secret
	kubectl apply -f deploy/k8s/
	kubectl -n scholarrag rollout status deploy/api --timeout=180s

k8s-status: ## Show pods, services, and the ingress
	kubectl -n scholarrag get pods,svc,ingress

k8s-seed: ## Run the in-cluster corpus seed Job (needs a corpus-baked image loaded)
	kubectl -n scholarrag delete job seed --ignore-not-found
	kubectl -n scholarrag apply -f deploy/k8s/seed-job.yaml
	kubectl -n scholarrag wait --for=condition=complete job/seed --timeout=600s
	kubectl -n scholarrag logs job/seed --tail=20

cluster-down: ## Delete the kind cluster
	kind delete cluster --name scholarrag

# ── Helm (Phase 7 Step 3) ───────────────────────────────────────────────────
helm-lint: ## Lint the chart for errors
	helm lint deploy/helm/scholarrag

helm-template: ## Render the chart to stdout (no cluster needed — great for learning)
	helm template scholarrag deploy/helm/scholarrag --namespace scholarrag

helm-deploy: ## Install/upgrade the release (creates namespace + secret first)
	kubectl create namespace scholarrag --dry-run=client -o yaml | kubectl apply -f -
	$(MAKE) k8s-secret
	helm upgrade --install scholarrag deploy/helm/scholarrag --namespace scholarrag --wait --timeout 5m

helm-uninstall: ## Remove the Helm release
	helm uninstall scholarrag --namespace scholarrag

# ── Terraform (Phase 7 Step 4) ──────────────────────────────────────────────
# tf-plan/apply read the Gemini key straight from .env into TF_VAR_gemini_api_key.
tf-init: ## Download providers + initialize the Terraform working dir
	terraform -chdir=deploy/terraform init

tf-validate: ## Check the config is syntactically + internally valid
	terraform -chdir=deploy/terraform validate

# Both secret keys, read from .env, passed as TF_VAR_* (Terraform masks them).
TF_KEYS = TF_VAR_gemini_api_key="$$(grep -E '^GEMINI_API_KEY=' .env | cut -d= -f2-)" TF_VAR_pinecone_api_key="$$(grep -E '^PINECONE_API_KEY=' .env | cut -d= -f2-)"

tf-plan: ## Preview what apply would create (no changes made)
	$(TF_KEYS) terraform -chdir=deploy/terraform plan

tf-apply: ## Provision the cluster AND deploy the app — one command, from nothing
	# TWO-STEP by necessity: the kubernetes/helm providers are configured FROM the
	# kind_cluster's outputs, which are unknown until it exists. So create the
	# cluster first (-target), then apply the rest against its now-known endpoint.
	$(TF_KEYS) terraform -chdir=deploy/terraform apply -target=kind_cluster.default -auto-approve
	$(TF_KEYS) terraform -chdir=deploy/terraform apply -auto-approve

tf-destroy: ## Tear the whole thing down (cluster + app)
	terraform -chdir=deploy/terraform destroy
