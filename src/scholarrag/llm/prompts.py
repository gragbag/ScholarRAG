"""Prompt templates for LLM calls — kept as data, separate from logic.

Keeping prompts here (not inline in the code that calls the LLM) makes them easy
to read, tweak, and version without touching control flow. Step 4 will add the
grounded-generation prompt alongside these.
"""

from __future__ import annotations

QUERY_REWRITE_SYSTEM = (
    "You are a query-expansion assistant for a document retrieval system. "
    "You rewrite a user's question into alternative search queries that surface "
    "relevant passages, expanding acronyms and adding synonyms. You output only "
    "the queries, one per line, with no numbering, commentary, or preamble."
)


def render_query_rewrite_prompt(query: str, num_variations: int) -> str:
    """Build the user-turn prompt asking for ``num_variations`` rewritten queries."""
    return (
        f"Produce {num_variations} alternative search queries for the question "
        f"below. Vary the wording, expand any acronyms, and add likely synonyms, "
        f"while preserving the original intent. Output one query per line.\n\n"
        f"Question: {query}"
    )
