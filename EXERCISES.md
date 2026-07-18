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

---

## When you're done

Show me your diffs (or just say "done") and I'll review before the next step.
