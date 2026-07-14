# ScholarRAG

A production-style **Retrieval-Augmented Generation** platform. Upload a corpus
of documents (default: scientific papers / research PDFs), ask natural-language
questions, and get grounded, cited answers.

This is deliberately not a toy "chat with PDF". It targets robust retrieval
(hybrid dense + lexical with reranking and query rewriting), an automated
evaluation harness wired into CI, LLM observability, experiment tracking,
guardrails, and a cloud-native deployment path. The document domain is swappable
via a corpus config module.

> **Status:** Phase 0 (scaffold & hygiene). See [Roadmap](#roadmap).

## Architecture (target)

```
                         ┌──────────────┐
        upload ───────►  │  FastAPI API │  ◄─────── query (streaming)
                         └──────┬───────┘
                                │ enqueue           │ retrieve + generate
                     ┌──────────▼─────────┐         ▼
                     │  Ingestion workers │   ┌──────────────────────────────┐
                     │ parse→chunk→embed  │   │ Retrieval pipeline           │
                     │ retries/DLQ/idemp. │   │ dense (Pinecone) + BM25 (PG) │
                     └──────────┬─────────┘   │ →RRF →cross-encoder rerank    │
                    dense │     │ metadata     │ →(query rewrite/HyDE)        │
              ┌───────────▼──┐  ▼   +BM25 FTS  │ →LLM answer + citations      │
              │  Pinecone    │ ┌────────┐      └───────────────┬──────────────┘
              │ (serverless) │ │Postgres│◄── lexical search    │
              └──────────────┘ └────────┘      ┌───────┐       │
                                                │ Redis │◄─ semantic cache
                                                └───────┘
   Cross-cutting: Langfuse (tracing+cost) │ MLflow (eval experiments) │
                  Prometheus/Grafana │ guardrails │ rate limiting
```

Because Pinecone is a managed cloud service, it sits behind a `VectorStore`
protocol with a `LocalVectorStore` fallback used automatically in tests/CI when
no Pinecone key is set — so local dev and CI stay hermetic and free.

## Tech stack

Python 3.12 · FastAPI · Pydantic v2 · **Claude** (Haiku for cheap-tier query
transforms, Sonnet for grounded generation with citations — provider-agnostic,
swappable via env var) · `sentence-transformers` (BGE) embeddings · Pinecone
(serverless) with local fallback · Postgres (metadata + BM25) · Redis · Celery ·
RAGAS · MLflow · Langfuse · Docker Compose · uv.

See [docs/DESIGN.md](docs/DESIGN.md) for decisions and tradeoffs.

## Getting started

Requires [uv](https://docs.astral.sh/uv/), Docker, and Docker Compose.

```bash
cp .env.example .env         # fill in ANTHROPIC_API_KEY when you reach Phase 2
make install                 # create the venv and install deps (uv sync)
make test                    # run the test suite (uses LocalVectorStore, no cloud)
make lint                    # ruff + mypy
make run                     # start the API locally (http://localhost:8001/docs)
make up                      # boot the full stack via docker-compose
```

Health check: `curl localhost:8001/health` · Config: `curl localhost:8001/info`

> Host ports are offset so ScholarRAG coexists with other local stacks: API
> **8001**, MLflow **5001**, Langfuse **3001**, Postgres **5433**, Redis **6380**.

## Roadmap

Phases 0–4 are the MVP; 5–7 are extensions.

- [x] **Phase 0** — Scaffold & hygiene (this phase)
- [ ] **Phase 1** — Async ingestion pipeline (parse → chunk → embed → index)
- [ ] **Phase 2** — Hybrid retrieval + grounded, cited generation
- [ ] **Phase 3** — Evaluation harness (retrieval metrics + RAGAS, CI gate, MLflow)
- [ ] **Phase 4** — Observability, semantic cache, guardrails
- [ ] **Phase 5** — Agentic RAG (LangGraph)
- [ ] **Phase 6** — Streamlit chat UI
- [ ] **Phase 7** — Kubernetes + Helm + Terraform deploy

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — decisions & tradeoffs
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — real measured numbers
- [docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md) — résumé bullets backed by benchmarks
