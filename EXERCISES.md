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

## When you're done

Show me your diffs (or just say "done") and I'll review — style, edge cases,
anything I'd flag in a real code review — before we move on to Phase 1.
