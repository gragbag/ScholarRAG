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

---

## When you're done

Show me your diffs (or just say "done") and I'll review before the next step.
