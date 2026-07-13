"""Corpus configuration — the swappable document-domain module.

The platform defaults to scientific papers / research PDFs, but the whole point
is that the domain is *configuration*, not code. A :class:`CorpusProfile`
bundles everything domain-specific — chunking parameters, the accepted file
types, and the prompt framing used for grounded generation — behind a name.

Select a profile with the ``CORPUS_PROFILE`` env var. Add a new domain by
registering another :class:`CorpusProfile`; nothing else in the pipeline needs
to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    """Everything domain-specific about a corpus, in one place."""

    name: str
    description: str
    # Chunking (token-ish word counts; the concrete splitter arrives in Phase 1).
    chunk_size: int = 800
    chunk_overlap: int = 120
    # Accepted source extensions (lower-case, with leading dot).
    file_types: tuple[str, ...] = (".pdf", ".md", ".txt")
    # System-prompt framing for grounded generation (Phase 2).
    answer_system_prompt: str = (
        "You are a careful research assistant. Answer the question using ONLY "
        "the provided source excerpts. Cite the excerpts you rely on. If the "
        "sources do not contain the answer, say so plainly."
    )
    # Free-form knobs a domain may want without changing the dataclass.
    extra: dict[str, str] = field(default_factory=dict)


_RESEARCH_PAPERS = CorpusProfile(
    name="research_papers",
    description="Scientific papers and research PDFs (the default domain).",
    chunk_size=900,
    chunk_overlap=150,
    file_types=(".pdf", ".md", ".txt"),
    answer_system_prompt=(
        "You are a meticulous research assistant answering questions about a "
        "corpus of scientific papers. Answer using ONLY the provided source "
        "excerpts and cite each excerpt you rely on. Prefer precise, technical "
        "language. If the excerpts do not support an answer, say so rather than "
        "speculating."
    ),
)

_GENERIC_DOCS = CorpusProfile(
    name="generic_docs",
    description="Any mixed corpus of PDF / markdown / text documents.",
    chunk_size=800,
    chunk_overlap=120,
)

_REGISTRY: dict[str, CorpusProfile] = {
    _RESEARCH_PAPERS.name: _RESEARCH_PAPERS,
    _GENERIC_DOCS.name: _GENERIC_DOCS,
}


def available_profiles() -> tuple[str, ...]:
    """Names of all registered corpus profiles."""
    return tuple(_REGISTRY)


def get_corpus_profile(name: str) -> CorpusProfile:
    """Look up a corpus profile by name.

    Raises:
        KeyError: if no profile is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - trivial
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown corpus profile {name!r}; known profiles: {known}") from exc
