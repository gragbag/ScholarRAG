"""Generate a synthetic eval set with the LLM (offline; `make eval-gen`).

Reads each file in the sample corpus, asks the *cheap* LLM tier for a few
questions answerable only from it (plus a one-line answer), and labels each with
that filename. Writes ``data/eval/synthetic.json``. Costs a few tokens once; the
eval run itself stays LLM-free.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from scholarrag.config import get_settings
from scholarrag.ingestion.parse import (
    UnsupportedFileTypeError,
    detect_content_type,
    extract_text,
)
from scholarrag.llm import build_llm_client

SAMPLE_CORPUS = Path(__file__).resolve().parents[3] / "data" / "sample_corpus"
OUT = Path(__file__).resolve().parents[3] / "data" / "eval" / "synthetic.json"
NUM_PER_DOC = 3
MAX_CHARS = 12_000  # cap the text fed to the LLM (abstract + intro is plenty for Qgen)

_SYSTEM = (
    "You write evaluation questions for a retrieval system. Given a document, you "
    "produce questions answerable ONLY from it, each with a one-sentence answer. "
    "Output one 'question | answer' per line, no numbering or commentary."
)
_NUMBER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


def main() -> None:  # pragma: no cover - manual entry point
    settings = get_settings()
    llm = build_llm_client(settings)
    examples: list[dict[str, object]] = []

    for path in sorted(SAMPLE_CORPUS.iterdir()):
        if not path.is_file():
            continue
        # Extract clean text through the ingestion parser (handles PDF, not just
        # text) instead of reading raw bytes.
        try:
            text = extract_text(path.read_bytes(), detect_content_type(path.name))
        except UnsupportedFileTypeError:
            continue
        prompt = (
            f"Document ({path.name}):\n{text[:MAX_CHARS]}\n\n"
            f"Write {NUM_PER_DOC} 'question | answer' lines for this document."
        )
        raw = llm.complete(prompt, system=_SYSTEM, tier="cheap")
        for line in raw.splitlines():
            if "|" not in line:
                continue
            question, answer = line.split("|", 1)
            question = _NUMBER_RE.sub("", question).strip()
            if question:
                examples.append(
                    {
                        "question": question,
                        "relevant_files": [path.name],
                        "reference_answer": answer.strip(),
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False keeps Greek letters and curly quotes readable (not \uXXXX).
    OUT.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(examples)} synthetic examples to {OUT}")


if __name__ == "__main__":  # pragma: no cover
    main()
