"""Output guardrails — the grounding gate.

The eval harness *measures* faithfulness offline; this gate *enforces* it per
request: an answer that cites no sources is either an honest refusal (fine) or a
hallucination dressed as an answer (not fine) — the gate tells them apart and
replaces the latter with a clean refusal.
"""

from __future__ import annotations

from scholarrag.generation.base import Answer

# What the user gets instead of an ungrounded answer.
REFUSAL_MESSAGE = (
    "I couldn't ground an answer to that question in the indexed documents, "
    "so I'd rather not guess. Try rephrasing, or ingest documents that cover it."
)

# Phrases that mark a *legitimate* "the sources don't cover this" answer — the
# system prompt instructs the model to say so, and those must pass the gate.
_REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "don't know",
    "do not know",
    "cannot answer",
    "can't answer",
    "not contain",  # "the sources do not contain..."
)


def looks_like_refusal(text: str) -> bool:
    """Heuristic: does this answer admit the sources don't cover the question?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def enforce_grounding(answer: Answer) -> Answer:
    "Pass grounded answers through; replace ungrounded assertions with a refusal."

    if answer.sources or looks_like_refusal(answer.text):
        return answer

    return Answer(text=REFUSAL_MESSAGE, sources=[])
