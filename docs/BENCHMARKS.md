# Benchmarks

**Real, measured numbers only.** Never invent a value here — every figure must be
reproducible and defensible. Each entry records the baseline, the method, the
date, and the hardware/config so the number means something.

## Environment

| Field | Value |
|---|---|
| Recorded on | 2026-07-20 → 2026-07-22 |
| CPU / RAM | Intel Core Ultra 7 265F (20 threads) / 16 GB — WSL2 |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, local CPU) |
| LLM (cheap / strong) | `gemini-3.1-flash-lite` / `gemini-3.5-flash` (free tier; provider swappable via `LLM_PROVIDER`) |
| Vector store | Pinecone serverless (AWS us-east-1) + Postgres FTS (lexical) |
| Corpus | 11 documents (8 arXiv PDFs + 3 md/txt), 253 chunks |

## Results

### Retrieval quality

Method: `make eval` — hand-rolled Recall@k / MRR / nDCG@k over the labelled
eval set (13 hand-written golden + 31 LLM-generated synthetic = 44 queries),
document-level relevance keyed by filename, k=5.

| Config | recall@5 | precision@5 | MRR | nDCG@5 | Method / date |
|---|---|---|---|---|---|
| hybrid (dense+lexical+RRF), no rerank, no rewriting | **0.977** | 0.195 | **0.905** | **0.924** | `make eval`, 2026-07-22 |
| hybrid + cross-encoder rerank | _pending_ | | | | flip `RERANKER_PROVIDER=cross_encoder` |
| + query rewriting (multi-query) | _pending_ | | | | flip `QUERY_REWRITING_ENABLED=true` |

Notes: every query has exactly **one** relevant document, so precision@5 is
capped at 1/5 = 0.2 — the measured 0.195 means the relevant doc was found in the
top-5 for ~98% of queries (it mirrors recall, not noise). MRR 0.905 = the right
document ranks #1 for the large majority of queries.

### Generation quality (RAGAS)

Method: `make eval-rag` — full pipeline over 8 eval examples; judge =
`gemini-3.1-flash-lite` + BGE embeddings via LangChain wrappers; logged to
MLflow (experiment `scholarrag-generation-eval`).

| Config | Faithfulness | Answer relevancy | Context precision | Context recall | Date |
|---|---|---|---|---|---|
| **hand-rolled** pipeline, no rerank, no rewriting, gen=`gemini-3.5-flash` | **0.979** | 0.932 | 0.937 | 1.000 | 2026-07-20 |
| **LangChain (LCEL)** pipeline, same config (`PIPELINE=langchain`) | **0.979** | 0.929 | 0.929 | 1.000 | 2026-07-23 |

Note: LLM-as-judge scores are directional, not gospel (judge noise is a known
limitation); faithfulness 0.979 over 8 examples ≈ answers are almost entirely
supported by retrieved context.

#### Pipeline A/B verdict (hand-rolled vs LangChain/LCEL)

Same retriever object, same prompts, same model config in both pipelines — the
A/B isolates the orchestration/generation glue. Findings (2026-07-23):

- **Quality parity.** Identical faithfulness (0.979); deltas of 0.003–0.008 on
  the other metrics over 8 examples are within judge noise. A live side-by-side
  query cited the same source with equivalent answers in both pipelines.
- **Latency:** not distinguishable — retrieval is shared by construction and
  Gemini free-tier variance (±6 s) dwarfs the glue layer. Single-sample timings
  (31.9 s vs 17.3 s) reflect cold-start asymmetry, not the framework.
- **What LangChain bought:** streaming came free from `chain.stream` (the
  hand-rolled path needed protocol + Answerer methods for the same feature);
  the generation glue is fewer lines.
- **What it cost:** cross-cutting policy (citation mapping, grounding gate,
  cache-aside) had to be consciously re-attached to the second pipeline; the
  LangChain path is currently **invisible to Langfuse** (it bypasses our
  instrumented `GeminiLLM` — fixing it needs LangChain's callback handler); and
  the dependency is pinned to the langchain 0.3 line by RAGAS compatibility.
- **Decision (2026-07-23): `langchain` is now the default pipeline** — quality
  parity being established, the LCEL path was promoted as the on-ramp to the
  Phase 5 agentic (LangGraph) work. The hand-rolled baseline stays toggleable
  (`PIPELINE=handrolled`) for comparison. Known gap inherited by the new
  default: LC-path generations are not yet visible in Langfuse (callback
  handler planned with Phase 5).

### Latency (by stage) & tokens

Method: OpenTelemetry traces in Jaeger (`POST /query` spans), Langfuse
generations for token counts. Measured 2026-07-22 against the live stack.

| Stage | Cold (first request) | Warm | Notes |
|---|---|---|---|
| `retrieve.dense` (BGE + Pinecone) | 6.70 s | **0.56–0.73 s** | cold = one-time BGE model load (~130 MB) |
| `retrieve.lexical` (Postgres FTS) | 24 ms | **9–10 ms** | actual `SELECT` ≈ 4.6 ms |
| generation (`gemini-3.5-flash`) | — | **≈ 6–12 s** | dominates total; high variance on free tier |
| `POST /query` total | 16.4 s | 7.3–13.3 s | |

Tokens per query (Langfuse, one representative query): **2,483 in / 45 out** —
RAG is input-dominated (~55:1), so `top_k` × chunk size is the main cost lever.

### Caching (exact + semantic answer cache, Redis)

Method: in-process timing of `QueryEngine.query`, same stack as the API,
2026-07-22. Threshold 0.93 (cosine, BGE).

| Scenario | Latency | LLM tokens spent |
|---|---|---|
| cache miss (full pipeline, cold) | 32.6 s | ~2.5 K |
| exact repeat of the same query | **< 1 ms** | 0 |
| paraphrase (semantic hit) | **6 ms** | 0 |

All three returned the identical answer. A cache hit removes ~100% of token
cost and >99.9% of latency; `make seed` invalidates the cache on corpus change.

### Agentic vs single-shot (the hard set)

Method: `make eval-agentic` — both pipelines over `data/eval/hard.json`
(8 answerable: oblique rephrasings + multi-hop; 3 unanswerable refusal-controls).
Deterministic metrics, no LLM judge (binary hypotheses; judge noise would swamp
n=11). Hypotheses: H1 agentic recovers refusals on hard answerable questions;
H2 false-answer rate on controls stays 0 for both; H3 the cost multiplier.

**Generation model:** `gemini-3.1-flash-lite` (cheap tier), *not* the `gemini-3.5-flash`
used elsewhere in this doc. Reason: the strong model is free-tier-capped at 20
requests/day and a full run needs ~22 generate calls — it cannot complete on free
tier. Both pipelines use flash-lite identically, so the A-vs-B comparison is fair;
only cross-comparison to the RAGAS rows above (which used flash) is affected.

| Pipeline | answered (answerable) | false answers (controls) | source-hit | calls/query | latency | Date |
|---|---|---|---|---|---|---|
| langchain (single-shot) | _pending_ | | | | | |
| agentic (grade+retry) | _pending_ | | | | | |

### Ingestion throughput

| Config | docs/min | Date |
|---|---|---|
| _pending_ — measure with the Celery worker over the 8-PDF corpus | | |
