"""Anthropic (Claude) LLM client — production default.

Wraps the Anthropic Messages API. The SDK is lazy-imported (it lives in the
``llm`` extra, absent in CI — same pattern as BGE/Pinecone), so importing this
module never requires the package or a key. ``FakeLLM`` is the tests/CI backend.

``create_fn`` is a test seam (like the reranker's ``predict_fn``): leave it
``None`` in real use — the client is loaded on demand; tests inject a stub so the
``complete`` logic can be checked without the SDK, a key, or a network call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings
from scholarrag.llm.base import ModelTier

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import Anthropic

# Signature of anthropic's ``client.messages.create`` (only the kwargs we pass).
# Returns the SDK's Message object; typed Any so tests can hand back a stub.
CreateFn = Callable[..., Any]
# Test seam for streaming: given the same kwargs, yield text deltas.
StreamFn = Callable[..., Iterable[str]]


class AnthropicLLM:
    """Claude-backed :class:`LLMClient`. Resolves tier -> model from settings."""

    def __init__(
        self,
        settings: Settings,
        *,
        create_fn: CreateFn | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        if settings.anthropic_api_key is None and create_fn is None and stream_fn is None:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicLLM")
        self._settings = settings
        self._create_fn = create_fn
        self._stream_fn = stream_fn
        self._client: Anthropic | None = None

    def _client_lazy(self) -> Anthropic:
        """Construct (and cache) the Anthropic SDK client on first real use."""
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def _create(self, **kwargs: Any) -> Any:
        """Call the Messages API — via the injected stub or the real SDK."""
        if self._create_fn is not None:
            return self._create_fn(**kwargs)
        return self._client_lazy().messages.create(**kwargs)

    def _model_for_tier(self, tier: ModelTier) -> str:
        """Map the semantic tier to a concrete model id from settings."""
        if tier == "strong":
            return self._settings.llm_model_strong
        return self._settings.llm_model_cheap

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "cheap",
        max_tokens: int | None = None,
    ) -> str:
        "Send ``prompt`` to Claude and return the text of the response."

        model = self._model_for_tier(tier)
        max_tokens = max_tokens or self._settings.llm_max_output_tokens

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            # Disable extended thinking: query rewriting and grounded generation
            # are straightforward tasks, and on Sonnet 5 adaptive thinking is on by
            # default and shares the max_tokens budget with the answer (risking
            # truncation). Off = predictable latency/cost. (Sonnet 5 / Haiku 4.5
            # both accept "disabled"; only Fable 5 would reject it.)
            "thinking": {"type": "disabled"},
        }

        if system is not None:
            kwargs["system"] = system

        response = self._create(**kwargs)
        return "".join(b.text for b in response.content if b.type == "text")

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "strong",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream Claude's response as text deltas (used by the SSE endpoint)."""
        kwargs: dict[str, Any] = {
            "model": self._model_for_tier(tier),
            "max_tokens": max_tokens or self._settings.llm_max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},  # see complete() for why
        }
        if system is not None:
            kwargs["system"] = system

        if self._stream_fn is not None:
            yield from self._stream_fn(**kwargs)
            return
        # The SDK's messages.stream is a context manager exposing a text-delta
        # iterator; yield each delta as it arrives.
        with self._client_lazy().messages.stream(**kwargs) as stream:
            yield from stream.text_stream
