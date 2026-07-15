"""Tests for the chunker.

The dataclass test passes now; the behaviour tests are the Step 3 exercise —
remove each ``@pytest.mark.skip`` once you've implemented ``chunk_text``. They
pin down the exact windowing behaviour (sizes, overlap, no trailing duplicate).
"""

from __future__ import annotations

import pytest

from scholarrag.corpus import CorpusProfile
from scholarrag.ingestion import TextChunk, chunk_text


def _profile(size: int, overlap: int) -> CorpusProfile:
    return CorpusProfile(name="t", description="test", chunk_size=size, chunk_overlap=overlap)


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_textchunk_dataclass() -> None:
    c = TextChunk(index=0, text="hello world", char_count=11)
    assert (c.index, c.text, c.char_count) == (0, "hello world", 11)


def test_chunk_empty_text_returns_no_chunks() -> None:
    assert chunk_text("   ", _profile(4, 1)) == []


def test_chunk_short_text_is_single_chunk() -> None:
    chunks = chunk_text("one two", _profile(4, 1))
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "one two"


def test_chunk_sliding_window_with_overlap() -> None:
    # 10 words, size 4, overlap 1 -> stride 3 -> 3 windows sharing 1 word each.
    chunks = chunk_text(_words(10), _profile(4, 1))
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunk_no_trailing_duplicate_window() -> None:
    # 6 words, size 3, overlap 1 -> stride 2; last window is short, not padded.
    chunks = chunk_text(_words(6), _profile(3, 1))
    assert [c.text for c in chunks] == ["w0 w1 w2", "w2 w3 w4", "w4 w5"]


def test_chunk_char_count_matches_text() -> None:
    chunks = chunk_text(_words(10), _profile(4, 1))
    assert all(c.char_count == len(c.text) for c in chunks)
