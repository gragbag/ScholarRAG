"""LLM package.

Prompt -> text, behind an :class:`LLMClient` protocol with swappable backends:

* :class:`AnthropicLLM` — Claude via the Anthropic Messages API (default).
* :class:`GeminiLLM`    — Google Gemini (free-tier friendly).
* :class:`OllamaLLM`    — a local model via Ollama (free, unlimited, offline).
* :class:`FakeLLM`      — deterministic, dependency-free (tests / CI).

Callers request a semantic *tier* (``"cheap"`` / ``"strong"``); the client maps
it to a model. Use :func:`build_llm_client` to get the one implied by settings.
"""

from __future__ import annotations

from scholarrag.config import Settings, get_settings
from scholarrag.llm.anthropic import AnthropicLLM
from scholarrag.llm.base import LLMClient, LLMError, ModelTier
from scholarrag.llm.fake import FakeLLM
from scholarrag.llm.gemini import GeminiLLM
from scholarrag.llm.ollama import OllamaLLM
from scholarrag.llm.openai import OpenAILLM

__all__ = [
    "AnthropicLLM",
    "FakeLLM",
    "GeminiLLM",
    "LLMClient",
    "LLMError",
    "ModelTier",
    "OllamaLLM",
    "OpenAILLM",
    "build_llm_client",
]


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the LLM client implied by configuration (``LLM_PROVIDER``)."""
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "fake":
        return FakeLLM()
    if provider == "anthropic":
        return AnthropicLLM(settings)
    if provider == "gemini":
        return GeminiLLM(settings)
    if provider == "ollama":
        return OllamaLLM(settings)
    if provider == "openai":
        return OpenAILLM(settings)
    raise ValueError(f"unsupported LLM provider: {provider!r}")
