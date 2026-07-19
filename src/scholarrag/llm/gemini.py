"""Google Gemini LLM client — a free-tier-friendly alternative to Claude.

Same :class:`LLMClient` protocol as :class:`AnthropicLLM`, so switching is a
one-line config change (``LLM_PROVIDER=gemini``). The ``google-genai`` SDK is
lazy-imported (it lives in the ``llm`` extra, absent in CI), and ``generate_fn``
/ ``stream_fn`` are test seams — inject them to exercise the logic without the
SDK, a key, or a network call.

Gemini's SDK differs from Anthropic's: the prompt is ``contents`` (a plain
string), the system prompt and token cap go in a ``config`` object, and the
response text is just ``response.text``. Thinking is disabled the Gemini way —
``thinking_config.thinking_budget = 0`` — mirroring the Claude client.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings
from scholarrag.llm.base import ModelTier

if TYPE_CHECKING:  # pragma: no cover
    from google.genai import Client

# Test seams mirroring the SDK calls (model=, contents=, config=).
GenerateFn = Callable[..., Any]
StreamFn = Callable[..., Iterable[Any]]


class GeminiLLM:
    """Gemini-backed :class:`LLMClient`. Resolves tier -> model from settings."""

    def __init__(
        self,
        settings: Settings,
        *,
        generate_fn: GenerateFn | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        if settings.gemini_api_key is None and generate_fn is None and stream_fn is None:
            raise ValueError("GEMINI_API_KEY is required for GeminiLLM")
        self._settings = settings
        self._generate_fn = generate_fn
        self._stream_fn = stream_fn
        self._client: Client | None = None

    def _client_lazy(self) -> Client:
        """Construct (and cache) the google-genai client on first real use."""
        if self._client is None:
            from google.genai import Client

            self._client = Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _model_for_tier(self, tier: ModelTier) -> str:
        if tier == "strong":
            return self._settings.gemini_model_strong
        return self._settings.gemini_model_cheap

    def _config(self, system: str | None, max_tokens: int | None) -> dict[str, Any]:
        """Build the GenerateContentConfig dict (system prompt, token cap, no thinking)."""
        config: dict[str, Any] = {
            "max_output_tokens": max_tokens or self._settings.llm_max_output_tokens,
            "thinking_config": {"thinking_budget": 0},  # disable thinking (Gemini 2.5)
        }
        if system is not None:
            config["system_instruction"] = system
        return config

    def _generate(self, **kwargs: Any) -> Any:
        """One-shot generation — via the injected fn or the real SDK."""
        if self._generate_fn is not None:
            return self._generate_fn(**kwargs)
        return self._client_lazy().models.generate_content(**kwargs)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "cheap",
        max_tokens: int | None = None,
    ) -> str:
        "Send ``prompt`` to Gemini and return the text of the response."
        model = self._model_for_tier(tier)
        config = self._config(system, max_tokens)

        response = self._generate(model=model, contents=prompt, config=config)
        return response.text or ""

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "strong",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream Gemini's response as text deltas (used by the SSE endpoint)."""
        model = self._model_for_tier(tier)
        config = self._config(system, max_tokens)
        if self._stream_fn is not None:
            chunks: Iterable[Any] = self._stream_fn(model=model, contents=prompt, config=config)
        else:
            chunks = self._client_lazy().models.generate_content_stream(
                model=model, contents=prompt, config=config
            )
        for chunk in chunks:
            if chunk.text:
                yield chunk.text
