"""The evaluation dataset — questions labelled with the docs that answer them.

Relevance is keyed by **filename** (stable), not ``vector_id`` (whose
``document_id`` UUID changes on every re-seed). ``reference_answer`` is optional
and only used by the generation eval in Step 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One labelled question: which source files contain the answer."""

    question: str
    relevant_files: list[str]
    reference_answer: str | None = None


def load_examples(path: Path) -> list[EvalExample]:
    """Load a JSON array of ``{question, relevant_files, reference_answer?}`` objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalExample(
            question=item["question"],
            relevant_files=list(item["relevant_files"]),
            reference_answer=item.get("reference_answer"),
        )
        for item in data
    ]
