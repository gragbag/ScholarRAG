"""Tests for content hashing and text extraction (these pass now)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholarrag.ingestion import (
    UnsupportedFileTypeError,
    content_hash,
    detect_content_type,
    extract_text,
    extract_text_from_path,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── hashing ────────────────────────────────────────────────────────────────
def test_content_hash_is_deterministic() -> None:
    assert content_hash(b"hello world") == content_hash(b"hello world")


def test_content_hash_differs_for_different_bytes() -> None:
    assert content_hash(b"a") != content_hash(b"b")


def test_content_hash_is_sha256_hex() -> None:
    h = content_hash(b"")
    assert len(h) == 64
    # Known SHA-256 of the empty input.
    assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ── content-type detection ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("paper.pdf", "pdf"),
        ("notes.md", "md"),
        ("readme.markdown", "md"),
        ("data.txt", "txt"),
        ("SHOUTING.PDF", "pdf"),  # case-insensitive
    ],
)
def test_detect_content_type(name: str, expected: str) -> None:
    assert detect_content_type(name) == expected


def test_detect_unknown_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="unsupported file extension"):
        detect_content_type("archive.zip")


# ── extraction ──────────────────────────────────────────────────────────────
def test_extract_text_txt() -> None:
    assert extract_text(b"hello\nworld", "txt") == "hello\nworld"


def test_extract_text_md_keeps_content() -> None:
    out = extract_text(b"# Title\n\nbody text", "md")
    assert "# Title" in out
    assert "body text" in out


def test_extract_unsupported_content_type_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(b"whatever", "docx")


def test_extract_text_strips_control_bytes() -> None:
    # PDF glyph extraction emits NUL/control bytes, which Postgres text columns
    # reject; extraction must strip them while keeping tab/newline.
    out = extract_text(b"a\x00b\x01c\td\ne", "txt")
    assert "\x00" not in out and "\x01" not in out
    assert out == "abc\td\ne"  # tab and newline preserved


def test_extract_from_path_txt() -> None:
    out = extract_text_from_path(FIXTURES / "sample.txt")
    assert "Retrieval-augmented generation" in out


def test_extract_from_path_pdf() -> None:
    out = extract_text_from_path(FIXTURES / "sample.pdf").lower()
    assert "retrieval" in out
    assert "scholarrag" in out
