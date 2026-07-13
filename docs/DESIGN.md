# Design decisions & tradeoffs

Living document. Each phase appends the decisions it made and what was rejected.

## Guiding principles

- **Boring, free/OSS, well-supported tools.** No paid or always-on-cost
  dependency without explicit sign-off.
- **Hermetic tests and CI.** Nothing in the test path may require cloud
  credentials or a live LLM. This drives several choices below.
- **Every tool must be explainable in an interview.** No keyword-coverage cruft.

## Phase 0 decisions

### VectorStore behind a protocol, with a local fallback
Pinecone is a managed cloud service, not a container. We put it behind a small
`VectorStore` protocol (`upsert` / `query` / `delete` / `count`) with two
implementations: `PineconeVectorStore` (production) and `LocalVectorStore`
(in-process cosine search over NumPy). `build_vector_store()` returns the local
store whenever `PINECONE_API_KEY` is unset. This keeps CI free and
deterministic, and cleanly separates "the retrieval interface" from "the vendor".
*Rejected for now:* Chroma as the local fallback — an in-memory NumPy store is
zero-dependency and sufficient for tests; Chroma can be swapped in later if we
want a persistent local option.

### LLM provider: Claude by default, behind a provider-agnostic seam
The generation layer is provider-agnostic (swappable via `LLM_PROVIDER`), but the
default is Claude:
- **Model-tier split maps to the cost-control requirement.** Haiku 4.5 for the
  cheap tier (query rewriting / HyDE / classification), Sonnet 5 for the strong
  tier (final grounded generation).
- **Native citations.** Anthropic's Citations feature emits structured citations
  tied to source chunks — it implements the "grounded generation with citations"
  requirement directly rather than via prompt-engineering, which is both more
  reliable and a genuine differentiator.
- **No embeddings lock-in.** Anthropic has no embeddings API, so embeddings stay
  on local `sentence-transformers` (BGE) regardless — free and on-device. Claude
  is only ever the generation + query-transformation provider.

*What I'd reach for instead:* Gemini's free tier for zero-cost dev, or a cheap
OpenAI model. Both remain implementable behind the same interface. Ollama covers
the fully-offline case.

### uv for dependency management
Fast, reproducible (`uv.lock`), single tool for venv + install + run. Drives the
Dockerfile (multi-stage `uv sync --frozen`), Makefile, and CI.

### Why Pinecone at all, and what I'd use at scale
Pinecone's free Starter tier is enough to demonstrate a real managed vector DB
without cost. Its constraints are documented and handled: single region
(AWS `us-east-1`), and Starter indexes pause after ~3 weeks of inactivity (the
client will reconnect/wake gracefully — implemented in Phase 1/7). *At scale I'd
evaluate `pgvector`* (one less system, transactional with our metadata) *or
Qdrant* (open-source, self-hostable, rich filtering) depending on whether
operational simplicity or managed convenience mattered more.

## Open questions for later phases

- Hybrid lexical side: Postgres full-text (default, transparent, reuses PG) vs.
  Pinecone sparse-dense hybrid — to be benchmarked in Phase 3.
- Reranking: local cross-encoder (free, controllable) vs. Pinecone integrated
  rerank (free-tier capped). Defaulting to local; tradeoff recorded here.
