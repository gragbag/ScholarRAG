# Hands-on exercises

Three exercises against the Phase 0 code, easiest → meatiest. Each has a
pre-written failing test as your target (in `tests/test_exercises.py`), except
Exercise 3 which you'll add yourself (the reason is part of the lesson).

**The loop for every exercise:**
1. Delete the `@pytest.mark.skip(...)` line above the target test.
2. `make test` → watch it fail (**red**).
3. Implement until it passes (**green**).
4. `make lint` → keep ruff + mypy happy.

Run `make test` now to confirm your green starting point (you'll see the
exercise tests reported as *skipped*).

---

## Exercise 1 — Add a `legal_docs` corpus profile

**Concept:** *configuration over code.* A new document domain should be new
*data*, not new logic. You'll see that adding a whole "legal documents" mode
touches exactly one file and needs zero changes to the pipeline.

**File:** `src/scholarrag/corpus.py`
**Target test:** `test_legal_docs_profile_registered`

**Steps:**
1. In `corpus.py`, define a new module-level `CorpusProfile` (copy the shape of
   `_RESEARCH_PAPERS`). Give it `name="legal_docs"`, a legal-flavored
   `answer_system_prompt`, and whatever `chunk_size` / `file_types` you think fit
   contracts and case law.
2. Register it by adding it to the `_REGISTRY` dict.
3. Unskip the test and run `make test`.

**Hint:** `_REGISTRY` is keyed by `profile.name`, so registration is one line:
`_REGISTRY[_LEGAL_DOCS.name] = _LEGAL_DOCS` (or add it to the dict literal).

**Acceptance:** `test_legal_docs_profile_registered` passes; `make lint` clean.

**Stretch:** set `CORPUS_PROFILE=legal_docs` in your shell, run `make run`, and
hit `curl localhost:8000/info` — your profile should show up as the active one.

---

## Exercise 2 — Add a `GET /corpus/{name}` endpoint

**Concept:** FastAPI routing with a *path parameter*, a Pydantic *response
model*, and correct error handling (404 for unknown names).

**File:** `src/scholarrag/api/main.py`
**Target tests:** `test_get_corpus_endpoint`, then the stretch
`test_get_corpus_unknown_returns_404`.

**Steps:**
1. Add a Pydantic response model near the others, e.g.:
   ```python
   class CorpusProfileResponse(BaseModel):
       name: str
       description: str
       chunk_size: int
       chunk_overlap: int
       file_types: list[str]
   ```
2. Inside `create_app`, register a route (the existing endpoints show the
   pattern). A path parameter is just a function argument named to match the
   URL placeholder:
   ```python
   @app.get("/corpus/{name}", response_model=CorpusProfileResponse, tags=["corpus"])
   async def get_corpus(name: str) -> CorpusProfileResponse:
       ...
   ```
3. Look the profile up with `get_corpus_profile(name)` and map its fields onto
   the response model. Note `file_types` is a `tuple` on the dataclass but the
   model wants a `list` — convert it.
4. Unskip the first test; implement; go green. Then do the 404 stretch.

**Hint for the 404:** `get_corpus_profile` raises `KeyError` for unknown names.
Catch it and raise FastAPI's `HTTPException(status_code=404, detail=...)`.
You'll need `from fastapi import FastAPI, HTTPException`.

**Acceptance:** both endpoint tests pass; `make lint` clean; the new route shows
up in the interactive docs at `http://localhost:8000/docs`.

---

## Exercise 3 — Add `fetch(id)` to the `VectorStore` protocol

**Concept:** *evolving an interface and keeping implementations in sync.* This is
the big one. When you add a method to the `VectorStore` protocol, **mypy will
immediately fail** every implementation that doesn't have it — that's the type
system doing your bookkeeping. You'll feel why the protocol is worth having.

This exercise has no pre-written test in the repo, because a test that calls
`store.fetch(...)` wouldn't type-check until the method exists — so the very
first step has to be yours.

**Files:** `src/scholarrag/vectorstore/base.py`, `.../local.py`, `.../pinecone.py`

**Steps:**
1. Add `fetch` to the `VectorStore` protocol in `base.py`. Suggested signature —
   return the record's metadata, or `None` if the id isn't present:
   ```python
   def fetch(self, id: str, *, namespace: str = "") -> Metadata | None:
       """Return the metadata for ``id``, or None if it does not exist."""
       ...
   ```
2. Run `make lint`. **Watch mypy fail** — `LocalVectorStore` and
   `PineconeVectorStore` no longer satisfy the protocol. Read the errors; they're
   telling you exactly what's missing. (This is the lesson — savor it.)
3. Implement `fetch` in `LocalVectorStore` (`local.py`). The data lives in
   `self._ns(namespace)` as `id -> (vector, metadata)`.
4. Implement `fetch` in `PineconeVectorStore` (`pinecone.py`). The Pinecone
   client has an `index.fetch(ids=[...], namespace=...)` call; return the
   metadata from the result, or `None`. (It's fine to write this even though no
   test exercises it — mypy still requires the method to exist.)
5. Add this test to `tests/test_vectorstore.py` and make it pass:
   ```python
   def test_fetch_returns_metadata(store: LocalVectorStore) -> None:
       store.upsert([VectorRecord(id="a", values=[1.0, 0.0, 0.0], metadata={"doc": "1"})])
       assert store.fetch("a") == {"doc": "1"}

   def test_fetch_missing_returns_none(store: LocalVectorStore) -> None:
       assert store.fetch("nope") is None
   ```

**Acceptance:** both new tests pass; `make lint` clean (both implementations
satisfy the protocol again).

**Stretch:** the runtime `isinstance(store, VectorStore)` check in
`test_local_store_satisfies_protocol` still passes — why does it *not* verify
your new method's signature, only its name? (Look up how `@runtime_checkable`
protocols work.)

---

# Phase 1 exercises

## Step 1 — Implement `repository.list_documents`

**Concept:** SQLAlchemy 2.0 `select` — ordering + pagination.
**File:** `src/scholarrag/db/repository.py` · **Target test:** `test_list_documents`
in `tests/test_db.py` (needs Postgres running: `docker compose up -d postgres`).

Build a `select(Document)` statement with `.order_by(Document.created_at.desc())`,
`.limit(limit)`, `.offset(offset)`, then `return list(session.scalars(stmt).all())`.

## Step 2 — Implement `FakeEmbedder._embed`

**Concept:** a hashing bag-of-words embedding — turning text into a deterministic,
L2-normalized vector with no ML dependencies. Reinforces the cosine-similarity
idea from the vector store.
**File:** `src/scholarrag/embeddings/fake.py`
**Target tests:** the 5 skipped `test_fake_embed_*` tests in
`tests/test_embeddings.py` (remove each `@pytest.mark.skip` to turn them on).

**Steps:**
1. `import hashlib` at the top of the file.
2. In `_embed`: start with `self._dim` zeros; for each token from
   `_tokenize(text)`, hash it to a bucket `idx = int(hashlib.sha1(token.encode()).hexdigest(), 16) % self._dim`
   and add `1.0` to `vec[idx]`.
3. L2-normalize: divide every element by `sqrt(sum(x*x for x in vec))`. If that
   length is `0` (empty text), return the zeros unchanged — don't divide by zero.

**Why normalize?** Unit vectors make cosine similarity a plain dot product — the
same convention `LocalEmbedder` uses (`normalize_embeddings=True`). It's what
makes the `shared_words_more_similar` test meaningful.

**Acceptance:** all 5 `test_fake_embed_*` pass; `ruff check` + `mypy` clean.

## Step 3 — Implement `chunk_text` (the sliding window)

**Concept:** word-based chunking with overlap — the core retrieval-quality knob.
Watch out for the off-by-one that emits a trailing duplicate window.
**File:** `src/scholarrag/ingestion/chunk.py`
**Target tests:** the 5 skipped `test_chunk_*` tests in
`tests/test_ingestion_chunk.py`.

**Steps:**
1. `words = text.split()`; if empty, return `[]`.
2. `size = profile.chunk_size`, `overlap = profile.chunk_overlap`;
   `stride = size - overlap`.
3. Walk `start` from 0 upward by `stride`. Each window is `words[start:start+size]`,
   joined with spaces. Wrap as `TextChunk(index=i, text=..., char_count=len(text))`
   with `i` = 0, 1, 2, …
4. After appending, if `start + size >= len(words)`, **break** — otherwise you
   emit a trailing duplicate window (the off-by-one). The skipped
   `test_chunk_no_trailing_duplicate_window` catches exactly this.

**Worked example:** 10 words, size 4, overlap 1 (stride 3) → `["w0 w1 w2 w3",
"w3 w4 w5 w6", "w6 w7 w8 w9"]`.

**Acceptance:** all 5 `test_chunk_*` pass; `ruff check` + `mypy` clean.

## Step 4 — Implement `IngestionPipeline._build_records`

**Concept:** the "glue" that ties the layers together — mapping each chunk +
its embedding into a vector-store record *and* a DB chunk row, linked by a
deterministic `vector_id`.
**File:** `src/scholarrag/ingestion/pipeline.py`
**Target tests:** the 2 skipped `test_ingest_*` tests in
`tests/test_ingestion_pipeline.py` (need Postgres running).

**Steps:** `chunks` and `embeddings` are parallel lists. For each
`(chunk, embedding)` pair (use `zip(chunks, embeddings, strict=True)`):
1. `vector_id = f"{document_id}:{chunk.index}"`.
2. A `VectorRecord(id=vector_id, values=embedding, metadata={...})` with metadata
   `{"text": chunk.text, "document_id": str(document_id), "chunk_index":
   chunk.index, "filename": filename}`.
3. A `NewChunk(chunk_index=chunk.index, text=chunk.text, vector_id=vector_id,
   char_count=chunk.char_count)`.

Return `(records, new_chunks)`.

**Acceptance:** both `test_ingest_*` pass; `ruff check` + `mypy` clean.

## Step 5 — Implement `is_transient` (retry vs dead-letter)

**Concept:** a retry only helps if the failure might succeed next time. This
function is the "reliability brain" — it decides whether a failed ingestion task
is **retried** (transient) or sent to the **dead-letter queue** (permanent).
**File:** `src/scholarrag/workers/tasks.py`
**Target tests:** the 2 skipped `test_is_transient*` tests in
`tests/test_workers.py`.

**Steps:** implement the one-liner:
```python
return isinstance(exc, TRANSIENT_ERRORS)
```
`TRANSIENT_ERRORS` (defined just above) lists the retryable types
(`TransientIngestionError`, `ConnectionError`, `TimeoutError`). Everything else —
a corrupt file, an unsupported type, a bug — is permanent; retrying would just
loop, so it goes to the dead-letter state.

**Acceptance:** both `test_is_transient*` pass; `ruff check` + `mypy` clean.

## Step 6 — the document API (two parts)

### Exercise A — `GET /documents/{id}` status endpoint
**Concept:** the poll-for-status half of the async pattern — path param, 404, and
mapping a model to a response.
**File:** `src/scholarrag/api/routes/documents.py`
**Target test:** `test_get_document_status` in `tests/test_api_documents.py`
(needs Postgres).

Implement `get_document`: look up `repo.get_document(session, document_id)`; if
`None`, raise `HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")`;
otherwise `return _to_response(document)`. The `list_documents` route right above
shows the same mapping.

### Exercise B — `ingest_corpus` (the seed loop)
**Concept:** walk a directory and ingest each supported file, skipping the rest.
**File:** `src/scholarrag/scripts/seed.py`
**Target test:** `test_ingest_corpus` in `tests/test_api_documents.py`.

Implement `ingest_corpus`: iterate `sorted(corpus_dir.iterdir())`, keep files
(`path.is_file()`), read bytes, and `pipeline.ingest(session, data=..., filename=path.name,
profile=profile)`; wrap each in `try/except UnsupportedFileTypeError: continue` to
skip unsupported types; collect and return the `IngestResult`s.

**Acceptance:** both skipped tests pass; `ruff check` + `mypy` clean. Then, with
Postgres up and the embeddings extra installed, `make seed` populates the corpus.

# Phase 2 exercises

## Step 1 — the two retrievers (two functions)

### Exercise A — `DenseRetriever.retrieve` (semantic)
**File:** `src/scholarrag/retrieval/dense.py` · **Target:** `test_dense_retriever_ranks_by_meaning`.
Embed the query with `self._embedder.embed_query(query)`, `self._vector_store.query(vector, top_k=top_k)`,
then map each match → `RetrievedChunk` (id from `match.id`, the rest from
`match.metadata`, coercing types: `uuid.UUID(str(...))`, `int(...)`, `str(...)`).

### Exercise B — `LexicalRetriever.retrieve` (keyword, Postgres FTS)
**File:** `src/scholarrag/retrieval/lexical.py` · **Target:** `test_lexical_retriever_finds_keyword` (needs Postgres).
A SQLAlchemy full-text query: `func.websearch_to_tsquery("english", query)`,
`func.ts_rank(Chunk.fts, tsquery)`, `.where(Chunk.fts.op("@@")(tsquery))`,
`.order_by(rank.desc()).limit(top_k)`, join `Document` for the filename, then map
each `(chunk, rank, filename)` row → `RetrievedChunk` (id = `chunk.vector_id`).
Full step-by-step guidance is in each function's docstring.

**Acceptance:** both `test_*retriever*` pass; `ruff check` + `mypy` clean.

## Step 2 — hybrid retrieval: RRF fusion + reranking (three functions)

The two engines from Step 1 have complementary blind spots (dense = meaning,
lexical = exact terms). Step 2 runs both, **fuses** their ranked lists, then
**reranks** the shortlist for precision. All three targets are hermetic — no
Postgres, no torch.

### Exercise A — `reciprocal_rank_fusion` (the heart)
**File:** `src/scholarrag/retrieval/fusion.py` · **Target:** `tests/test_fusion.py`
(remove the module-level `pytestmark` skip).
Combine ranked lists by *position*, not score: sum `1/(k+rank)` (1-based) per
chunk id across lists; a chunk in both lists gets two contributions. Sort ids by
fused score desc, rebuild each `RetrievedChunk` with `replace(chunk, score=...)`
(it's frozen), apply `top_k`. Step-by-step in the docstring.

### Exercise B — `HybridRetriever.retrieve` (the composition)
**File:** `src/scholarrag/retrieval/hybrid.py` · **Target:** `tests/test_hybrid.py`.
Ask dense + lexical each for a `candidate_k` pool, `reciprocal_rank_fusion` them,
then: no reranker → `fused[:top_k]`; else `reranker.rerank(query, fused, top_k=top_k)`.

### Exercise C — `CrossEncoderReranker.rerank` (precision second stage)
**File:** `src/scholarrag/retrieval/rerank.py` · **Target:**
`test_cross_encoder_rerank_orders_by_score` in `tests/test_rerank.py`.
Build `(query, chunk.text)` pairs, score them all with `self._predict(pairs)`,
`replace` each chunk's score, sort desc, truncate. (`FakeReranker` is already
done for you as the test/CI backend; the test injects a stub `predict_fn` so you
never load a model.)

**Acceptance:** `test_fusion.py`, `test_hybrid.py`, and the cross-encoder test all
pass; `ruff check` + `mypy` clean. Flip `RERANKER_PROVIDER=cross_encoder` (with the
`embeddings` extra) to feel reranking on real queries.

## Step 3 — LLM provider layer + query rewriting (three functions)

First time the system talks to an LLM. An `LLMClient` protocol (Claude / Fake)
sits behind a semantic *tier* (`cheap`/`strong`), and query rewriting — the first
*use* of it — expands a question into several search queries before retrieval.
All three targets are hermetic (a stub `create_fn` / `FakeLLM`, no key/network).

### Exercise A — `AnthropicLLM.complete` (the Messages API)
**File:** `src/scholarrag/llm/anthropic.py` · **Target:**
`test_anthropic_complete_maps_tier_and_extracts_text` in `tests/test_llm.py`.
Resolve tier→model, build `messages=[{"role":"user","content":prompt}]` with
`system` as a separate top-level kwarg, call `self._create(**kwargs)`, and join
the `"text"` blocks of `response.content`. Do **not** pass `temperature` (Sonnet 5
400s on it). Step-by-step in the docstring.

### Exercise B — `QueryRewriter.rewrite`
**File:** `src/scholarrag/retrieval/rewrite.py` · **Target:**
`test_query_rewriter_includes_original_and_variations` in `tests/test_rewrite.py`.
Render the prompt, `self._llm.complete(..., tier="cheap")`, `parse_query_list` the
response, and return the original query first + deduped variations.

### Exercise C — `parse_query_list` (LLM-output wrangling)
**File:** `src/scholarrag/retrieval/rewrite.py` · **Target:**
`test_parse_query_list_cleans_and_dedupes` in `tests/test_rewrite.py`.
Split lines, strip `1. `/`2) `/`- `/`* ` numbering with a regex, drop blanks,
dedupe preserving order.

**Acceptance:** all three target tests pass; `ruff check` + `mypy` clean. Flip
`LLM_PROVIDER=anthropic` with a real key (and the `llm` extra) to run it live.

## Step 4a — grounded cited generation + query pipeline (three functions)

The payoff: turn retrieved chunks into a grounded, cited **answer**, and expose
it at `POST /query`. Everything composes — rewrite → multi-query retrieve → RRF
fuse → generate. All three targets are hermetic (`FakeLLM` + stub retriever).
Recommended order: **C → A → B** (B needs A and C).

### Exercise C — `extract_citations` (LLM-output wrangling)
**File:** `src/scholarrag/generation/citations.py` · **Target:**
`test_extract_citations` in `tests/test_generation.py`.
Regex out `[n]` markers, convert to ints, dedupe preserving order.

### Exercise A — `Answerer.answer` (grounded generation)
**File:** `src/scholarrag/generation/answerer.py` · **Target:**
`test_answerer_returns_cited_sources` in `tests/test_generation.py`.
Build the numbered-source prompt, `complete(..., tier="strong")`, `extract_citations`
the response, map cited numbers back to chunks, return `Answer(text, sources)`.

### Exercise B — `QueryEngine.query` (the capstone)
**File:** `src/scholarrag/pipeline/engine.py` · **Target:**
`test_query_engine_runs_full_flow` in `tests/test_pipeline.py`.
`rewriter.rewrite` → `retriever.retrieve` per query → `reciprocal_rank_fusion`
across the lists → `answerer.answer`. The whole system in ~4 lines.

**Acceptance:** all three target tests pass; `ruff check` + `mypy` clean. Then,
with `LLM_PROVIDER=anthropic` + a key (and Postgres/Pinecone seeded),
`curl -X POST localhost:8001/query -d '{"query":"..."}'` returns a cited answer.

# Phase 3 exercises

## Step 1 — retrieval metrics (three functions)

Hand-rolled, document-level IR metrics over a ranked list of distinct filenames
and the set of relevant filenames. `precision_at_k` is done as your template.
All hermetic (pure functions, no deps). Order: A → B → C (any order works).

### Exercise A — `recall_at_k`
**File:** `src/scholarrag/eval/retrieval_metrics.py` · **Target:** `test_recall_at_k`.
Fraction of relevant docs in the top-k: `len(set(ranked[:k]) & relevant) / len(relevant)`
(guard empty relevant → 0.0).

### Exercise B — `reciprocal_rank`
**File:** same · **Target:** `test_reciprocal_rank`.
Walk `ranked` 1-based; return `1 / position` at the first relevant hit, else 0.

### Exercise C — `ndcg_at_k`
**File:** same (add `import math`) · **Target:** `test_ndcg_at_k`.
`DCG@k / IDCG@k` with binary gain `1 / log2(i + 1)`; IDCG puts all relevant docs
at the top. Guard divide-by-zero.

**Acceptance:** the three metric tests pass; unskip `test_retrieval_eval_meets_recall_floor`
(the hermetic gate) and it passes too; `ruff check` + `mypy` clean. Then `make eval`
(with a seeded corpus) prints real Recall@k / MRR / nDCG for BENCHMARKS.md.

## Step 2 — generation eval with RAGAS + LangChain (two functions)

Measure *answer* quality (faithfulness, answer relevancy, context precision/recall)
with RAGAS, an LLM-as-judge framework that calls its judge through LangChain
interfaces. Both exercises are in `src/scholarrag/eval/ragas_eval.py`; import
`ragas`/`langchain` *inside* the functions (they live in the `eval` extra). Needs
`uv sync --extra eval`, a seeded corpus, and a Gemini key — run via `make eval-rag`.

### Exercise A — `build_judge` (LangChain)
Wrap Gemini (`ChatGoogleGenerativeAI`, cheap tier) and BGE (`HuggingFaceEmbeddings`,
`settings.embedding_model`) in their LangChain adapters, then in RAGAS's
`LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`. This is the LangChain moment:
LangChain is the provider-agnostic layer RAGAS talks through.

### Exercise B — `run_ragas_eval` (RAGAS)
Turn each `GenerationSample` into a `SingleTurnSample`
(`user_input` / `response` / `retrieved_contexts` / `reference`), wrap in an
`EvaluationDataset`, pick the four metric instances, call
`evaluate(dataset, metrics=, llm=, embeddings=, run_config=RunConfig(max_workers=))`,
and average each metric column from `result.to_pandas()`.

**Acceptance:** `make eval-rag` prints the four RAGAS scores and logs a run to
MLflow. (`collect_samples` is scaffolded and covered by `test_ragas_eval.py`.)
Then compare configs — rerank on/off, rewriting on/off — as MLflow runs.

# Phase 4 exercises

## Step 1 — Langfuse tracing (two instrumentation exercises)

Different flavour from earlier exercises: nothing to implement from scratch —
you *instrument* working code. The no-op-safe layer lives in
`src/scholarrag/observability/` (safe with no keys, no extra, no server).
Unskip each target test in `tests/test_observability.py` as you go.

### Exercise A — trace the pipeline
**File:** `src/scholarrag/pipeline/engine.py` · **Target:** `test_pipeline_stages_are_traced`.
Import `observe` from `scholarrag.observability` and decorate `query`
(`@observe(name="query")`), `answer_with_context` (`name="query-stream"`), and
`_retrieve` (`name="retrieve"`). Call nesting builds the trace tree.

### Exercise B — log the LLM generation with token usage
**File:** `src/scholarrag/llm/gemini.py` · **Target:** `test_gemini_reports_usage`.
Decorate `complete` with `@observe(name="gemini-complete", as_type="generation")`
and, after the response, call `update_current_generation(model=..., input=...,
output=..., usage={"input": prompt_tokens, "output": completion_tokens})` from
`response.usage_metadata` (guard `None`). Usage is what lights up cost in the UI.

**Acceptance:** both targets pass; `make check` clean. Then the live loop:
`docker compose up -d langfuse` → create account/project at localhost:3001 →
paste API keys into `.env` → `make run` → send a `/query` → watch the trace
(query → retrieve → gemini-complete, with tokens) appear in the Langfuse UI.

## Step 1b — OpenTelemetry app tracing (two exercises)

OTel sees what Langfuse can't: HTTP requests, SQL queries, the Celery hop.
The no-op-safe layer is `src/scholarrag/observability/otel.py`; spans ship to
Jaeger (`docker compose up -d jaeger`, UI on localhost:16686). Off until
`OTEL_EXPORTER_ENDPOINT` is set.

### Exercise A — the OTel setup ritual
**File:** `src/scholarrag/observability/otel.py` (`_setup_tracing`) · **Target:**
`test_configure_otel_enables_tracing`.
Resource (service name) → TracerProvider (set globally) → OTLP/HTTP exporter +
BatchSpanProcessor → instrumentors (FastAPI app if present, SQLAlchemy engine,
Celery). This same sequence is how you instrument *any* service — full guidance
in the docstring.

### Exercise B — manual spans: dense vs lexical
**File:** `src/scholarrag/retrieval/hybrid.py` · **Target:** `test_hybrid_emits_manual_spans`.
Fetch `get_tracer("scholarrag.retrieval")` *inside* `retrieve` (call time, not
import time) and wrap the dense/lexical sub-retrievals in
`tracer.start_as_current_span("retrieve.dense" / "retrieve.lexical")`. This is
the manual API — and it answers a real question: Pinecone vs Postgres latency.

**Acceptance:** both targets pass; `make check` clean. Live:
`docker compose up -d jaeger` → set `OTEL_EXPORTER_ENDPOINT=http://localhost:4318`
in `.env` → `make run` → send a `/query` → open localhost:16686, service
`scholarrag`, and inspect the trace: POST /query → retrieve.dense /
retrieve.lexical → the actual FTS `SELECT`.

## Step 2 — answer caching, exact + semantic (three exercises)

A cache hit skips retrieval AND the 6-12s LLM call. Two layers in Redis:
exact (hash of query+config) and semantic (BGE cosine over previously answered
queries, catches paraphrases). All hermetic — tests inject an in-memory
`FakeRedis`. Files: `src/scholarrag/cache/answer_cache.py` + `pipeline/engine.py`.

### Exercise A — the exact layer
**Target:** `test_exact_cache_hit_miss_and_config_isolation` in tests/test_cache.py.
`_exact_key`: SHA-256 of `f"{query}|{self._fingerprint}"` under `cache:exact:` —
the fingerprint keeps answers from one config (model/top_k/reranker) from ever
serving another. `_get_exact`: `redis.get` → `deserialize_answer` or None.

### Exercise B — the semantic layer
**Target:** `test_semantic_cache_hits_paraphrase_not_unrelated`.
Embed the query (`embed_query`), scan `cache:semantic:*` entries, cosine = dot
product (unit vectors), keep the best, return it only if `>= self._threshold`
(the threshold is the false-hit guard).

### Exercise C — cache-aside in the engine
**Target:** `test_query_engine_uses_cache`.
In `QueryEngine.query`: check `self._cache.get(query)` first (hit → return
immediately), else run the pipeline and `put` the answer. The standard
cache-aside pattern; guard for `self._cache is None`.

**Acceptance:** all three targets pass; `make check` clean. Then live:
`CACHE_ENABLED=true` in `.env`, restart `make run`, send the same query twice —
the second `POST /query` trace in Jaeger collapses from seconds to milliseconds
(no retrieve/generation spans), and Langfuse shows no second generation.
Numbers → BENCHMARKS (cost/latency saved per hit).

---

## When you're done

Show me your diffs (or just say "done") and I'll review before the next step.
