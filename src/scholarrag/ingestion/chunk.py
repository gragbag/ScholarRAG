"""Chunking — splitting a document's text into overlapping passages.

Why chunk? Embedding models have a hard token limit, and retrieval works far
better on focused passages than on a whole document embedded as one blurry
"average" vector. Chunk size is the single biggest retrieval-quality knob.

Why overlap? Consecutive chunks share some words so a fact that straddles a
boundary (setup at the end of one chunk, payoff at the start of the next) still
lives intact inside at least one chunk.

This is a word-based sliding window, configured by the corpus profile
(``chunk_size`` / ``chunk_overlap``, measured in words). A production system
might use a token-accurate or structure-aware splitter (e.g. LangChain's
``RecursiveCharacterTextSplitter``); we hand-roll a simple one here for clarity
and swap-ability — see docs/DESIGN.md for the tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholarrag.corpus import CorpusProfile


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One chunk of a document: its position, its text, and text length."""

    index: int  # 0-based order within the document
    text: str
    char_count: int


def chunk_text(text: str, profile: CorpusProfile) -> list[TextChunk]:
    """Split ``text`` into overlapping word-windows per the corpus ``profile``.

    ── YOUR TURN (Step 3 exercise) ─────────────────────────────────────────────
    Implement a word-based sliding window:

      1. Split ``text`` into words:  ``words = text.split()``.
         If there are no words, return ``[]`` (empty / whitespace-only input).
      2. Read ``size = profile.chunk_size`` and ``overlap = profile.chunk_overlap``
         (both are word counts; profiles guarantee ``overlap < size``).
      3. The window advances by a *stride* of ``size - overlap`` words each step.
         Walk ``start`` from 0 upward by ``stride``; each window is
         ``words[start : start + size]``. Join it back with spaces to form the
         chunk text, and wrap it as ``TextChunk(index=i, text=..., char_count=len(text))``
         with ``i`` counting 0, 1, 2, ...
      4. Stop once a window reaches the end: after appending, if
         ``start + size >= len(words)``, break — otherwise you'll emit a trailing
         duplicate window that's fully contained in the previous one (the classic
         off-by-one here).

    Worked example — 10 words, size=4, overlap=1 (stride=3) → 3 chunks:
      words[0:4], words[3:7], words[6:10]  (each shares 1 word with the next).

    Target tests: the ``@pytest.mark.skip``-ped ``test_chunk_*`` tests in
    tests/test_ingestion_chunk.py (remove the skips to start). See EXERCISES.md → Step 3.
    ────────────────────────────────────────────────────────────────────────────
    """
    words = text.split()
    if not words:
        return []
    
    size = profile.chunk_size
    overlap = profile.chunk_overlap
    stride = size - overlap

    start = 0
    i = 0
    chunks = []
    for i, start in enumerate(range(0, len(words), stride)):
        chunk_text = " ".join(words[start: start + size])
        chunks.append(TextChunk(index=i, text=chunk_text, char_count=len(chunk_text)))

        if start + size >= len(words):
            break

        start += stride

    return chunks

