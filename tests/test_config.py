"""Tests for settings resolution and the corpus config module."""

from __future__ import annotations

import pytest

from scholarrag.config import Settings
from scholarrag.corpus import available_profiles, get_corpus_profile


def test_defaults_prefer_claude_and_local(monkeypatch: pytest.MonkeyPatch) -> None:
    # This asserts the *code* defaults, so ignore both ambient env (CI exports
    # EMBEDDING_PROVIDER/VECTOR_STORE) and the developer's local .env
    # (which may set LLM_PROVIDER=gemini etc.).
    for var in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "VECTOR_STORE"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None, anthropic_api_key=None, pinecone_api_key=None)
    assert settings.llm_provider == "anthropic"
    assert settings.embedding_provider == "local"
    assert settings.use_pinecone is False


def test_use_pinecone_auto_follows_key_presence() -> None:
    assert Settings(vector_store="auto", pinecone_api_key=None).use_pinecone is False
    assert Settings(vector_store="auto", pinecone_api_key="pk-xxx").use_pinecone is True


def test_vector_store_override_wins_over_key() -> None:
    # Explicit "local" ignores a present key; explicit "pinecone" ignores absence.
    assert Settings(vector_store="local", pinecone_api_key="pk-xxx").use_pinecone is False
    assert Settings(vector_store="pinecone", pinecone_api_key=None).use_pinecone is True


def test_env_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("CORPUS_PROFILE", "generic_docs")
    settings = Settings()
    assert settings.llm_provider == "gemini"
    assert settings.corpus_profile == "generic_docs"


def test_corpus_profiles_registered() -> None:
    profiles = available_profiles()
    assert "research_papers" in profiles
    assert "generic_docs" in profiles


def test_default_corpus_profile_is_research_papers() -> None:
    profile = get_corpus_profile("research_papers")
    assert profile.name == "research_papers"
    assert ".pdf" in profile.file_types
    assert profile.chunk_size > profile.chunk_overlap


def test_unknown_corpus_profile_raises() -> None:
    with pytest.raises(KeyError, match="unknown corpus profile"):
        get_corpus_profile("does_not_exist")
