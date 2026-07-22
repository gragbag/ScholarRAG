"""Seed the vector store + database with a small sample corpus.

Runs the ingestion pipeline **synchronously** (no worker needed) so ``make seed``
just works. Uses the real configured embedder, so the vectors are real and
Phase 2 retrieval has something meaningful to search.

    make seed        # or: uv run python -m scholarrag.scripts.seed
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from scholarrag.cache import build_answer_cache
from scholarrag.config import get_settings
from scholarrag.corpus import CorpusProfile, get_corpus_profile
from scholarrag.db.engine import session_scope
from scholarrag.embeddings import build_embedder
from scholarrag.ingestion import IngestionPipeline, IngestResult
from scholarrag.ingestion.parse import UnsupportedFileTypeError
from scholarrag.vectorstore import build_vector_store

# repo-root/data/sample_corpus
SAMPLE_CORPUS = Path(__file__).resolve().parents[3] / "data" / "sample_corpus"


def ingest_corpus(
    pipeline: IngestionPipeline,
    session: Session,
    *,
    corpus_dir: Path,
    profile: CorpusProfile,
) -> list[IngestResult]:
    "Ingest every supported file in ``corpus_dir`` and return the results."
    results = []
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file():
            continue

        data = path.read_bytes()
        try:
            ingest_result = pipeline.ingest(
                session=session, data=data, filename=path.name, profile=profile
            )
            results.append(ingest_result)
        except UnsupportedFileTypeError:
            continue

    return results


def main() -> None:  # pragma: no cover - manual entry point
    settings = get_settings()
    embedder = build_embedder(settings)
    pipeline = IngestionPipeline(
        embedder=embedder,
        vector_store=build_vector_store(settings),
    )
    profile = get_corpus_profile(settings.corpus_profile)
    with session_scope() as session:
        results = ingest_corpus(pipeline, session, corpus_dir=SAMPLE_CORPUS, profile=profile)
    for r in results:
        print(
            f"{r.document_id}  status={r.status.value}  chunks={r.num_chunks}  skipped={r.skipped}"
        )
    print(f"seeded {len(results)} document(s) from {SAMPLE_CORPUS}")

    # A changed corpus makes cached answers stale — drop them (cache invalidation).
    cache = build_answer_cache(settings, embedder)
    if cache is not None:
        print(f"cleared {cache.clear()} cached answer(s)")


if __name__ == "__main__":  # pragma: no cover
    main()
