# Benchmarks

**Real, measured numbers only.** Never invent a value here — every figure must be
reproducible and defensible. Each entry records the baseline, the method, the
date, and the hardware/config so the number means something.

Metrics we will fill in as the phases land:

- **Retrieval quality:** recall@k, MRR, NDCG@k — before/after reranking, and
  before/after query rewriting (Phase 3).
- **Generation quality (RAGAS):** faithfulness, answer relevancy, context
  precision/recall (Phase 3).
- **Latency:** p50 / p95 per query, broken down by stage (Phase 4).
- **Cost:** median cost per query, and the % reduction from semantic caching +
  model-tier routing (Phase 4).
- **Ingestion throughput:** documents/min sustained by the worker pool (Phase 1).

## Environment

| Field | Value |
|---|---|
| Recorded on | _TBD_ |
| CPU / RAM | _TBD_ |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) |
| LLM (cheap / strong) | `claude-haiku-4-5` / `claude-sonnet-5` |
| Corpus | _TBD_ (N documents) |

## Results

_No measurements yet — Phase 0 is scaffolding only. Tables get populated from
Phase 1 onward._

### Retrieval metrics

| Config | recall@5 | MRR | NDCG@5 | Method / date |
|---|---|---|---|---|
| _pending_ | | | | |

### Generation quality (RAGAS)

| Config | Faithfulness | Answer relevancy | Context precision | Context recall | Date |
|---|---|---|---|---|---|
| _pending_ | | | | | |

### Latency & cost

| Config | p50 (ms) | p95 (ms) | Cost/query | Date |
|---|---|---|---|---|
| _pending_ | | | | |
