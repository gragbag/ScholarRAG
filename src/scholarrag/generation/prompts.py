"""Prompts for grounded, cited generation.

The system prompt is what *enforces grounding*: it tells the model to answer only
from the numbered sources and to say "I don't know" when they don't cover the
question — this is the instruction that turns a chatbot into a RAG system.
"""

from __future__ import annotations

from scholarrag.retrieval.base import RetrievedChunk

GROUNDED_SYSTEM = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "numbered sources provided below. Cite every claim with its source number in "
    "square brackets, e.g. [1] or [2]. If the sources do not contain the answer, "
    "say you don't have enough information to answer — never use outside knowledge "
    "or invent facts."
)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as a numbered source list the model can cite by number."""
    return "\n\n".join(
        f"[{i}] ({chunk.filename}) {chunk.text}" for i, chunk in enumerate(chunks, start=1)
    )


def render_answer_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the user-turn prompt: the numbered sources, then the question."""
    return f"Sources:\n{format_sources(chunks)}\n\nQuestion: {query}\n\nAnswer with citations:"
