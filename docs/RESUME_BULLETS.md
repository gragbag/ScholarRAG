# Résumé bullets

Format: **production metric > model metric > tools**, each backed by a real
number from [BENCHMARKS.md](BENCHMARKS.md). **Fill placeholders (X, Y, Z, A, B,
N, T) with true measured values only — never invent.**

Draft bullets (numbers land as the phases complete):

1. Built a production RAG platform (FastAPI, LangChain, Claude, Pinecone) with
   hybrid retrieval (dense + BM25 + Reciprocal Rank Fusion) and cross-encoder
   reranking, improving answer-relevance **NDCG@5 from X to Y** on an
   **N-document** corpus.

2. Cut median query cost **Z%** and p95 latency **from A ms to B ms** via a
   semantic answer cache and model-tier routing (Haiku for query rewriting,
   Sonnet for generation); instrumented token/cost/latency with Langfuse.

3. Engineered an async ingestion pipeline (Celery/Redis) with retries,
   idempotency (content-hash dedup), and a dead-letter queue, sustaining
   **T docs/min**; tracked retrieval-config experiments in MLflow.

4. Gated a RAGAS + retrieval-metric evaluation suite in CI (GitHub Actions),
   catching quality regressions before deploy; deployed on Kubernetes with
   Terraform-provisioned infrastructure.

5. Designed a provider-agnostic LLM/embedding layer and a `VectorStore`
   abstraction with a local fallback, keeping CI hermetic and free while
   supporting Pinecone in production.
