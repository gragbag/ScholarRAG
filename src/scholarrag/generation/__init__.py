"""Generation package — grounded, cited answers from retrieved chunks.

:class:`Answerer` composes a numbered-source prompt and the *strong* LLM tier
into an :class:`Answer`. Use :func:`build_answerer` to get one wired from settings.
"""

from __future__ import annotations

from scholarrag.config import Settings, get_settings
from scholarrag.generation.answerer import Answerer, cited_sources
from scholarrag.generation.base import Answer
from scholarrag.generation.citations import extract_citations
from scholarrag.llm import LLMClient, build_llm_client

__all__ = [
    "Answer",
    "Answerer",
    "build_answerer",
    "cited_sources",
    "extract_citations",
]


def build_answerer(settings: Settings | None = None, *, llm: LLMClient | None = None) -> Answerer:
    """Return an :class:`Answerer` backed by the configured LLM (injectable in tests)."""
    settings = settings or get_settings()
    llm = llm or build_llm_client(settings)
    return Answerer(llm=llm)
