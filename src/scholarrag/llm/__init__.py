"""LLM package.

Prompt -> text, behind an :class:`LLMClient` protocol with swappable backends:

* :class:`AnthropicLLM` — Claude via the Anthropic Messages API (default).
* :class:`FakeLLM`      — deterministic, dependency-free (tests / CI).

Callers request a semantic *tier* (``"cheap"`` / ``"strong"``); the client maps
it to a model. Use :func:`build_llm_client` to get the one implied by settings.
"""

from __future__ import annotations

from scholarrag.config import Settings, get_settings
from scholarrag.llm.anthropic import AnthropicLLM
from scholarrag.llm.base import LLMClient, LLMError, ModelTier
from scholarrag.llm.fake import FakeLLM

__all__ = [
    "AnthropicLLM",
    "FakeLLM",
    "LLMClient",
    "LLMError",
    "ModelTier",
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
    # gemini / openai / ollama — wired in a later phase.
    raise ValueError(f"unsupported LLM provider: {provider!r}")
