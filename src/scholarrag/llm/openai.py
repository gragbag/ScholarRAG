"""OpenAI-compatible LLM client — one class, many hosted providers.

The OpenAI Chat Completions API is the de-facto standard, so Groq, OpenRouter,
Together and DeepSeek all speak it. This single client targets any of them: keep
the code identical and point ``OPENAI_BASE_URL`` at the provider's host (plus its
model ids + key). That's how the deploy runs generation on a free hosted model
instead of Ollama or the rate-limited Gemini tier.

    Groq       -> https://api.groq.com/openai/v1  (llama-3.3-70b-versatile)
    OpenRouter -> https://openrouter.ai/api/v1  (deepseek/deepseek-chat)
    Together   -> https://api.together.xyz/v1  (meta-llama/Llama-3.3-70B-Instruct-Turbo)
    DeepSeek   -> https://api.deepseek.com  (deepseek-chat)

The ``openai`` SDK is lazy-imported (it lives in the ``llm`` extra, absent in
CI); ``create_fn`` / ``stream_fn`` are test seams mirroring the other clients, so
the logic is exercised without the SDK, a key, or a network call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings
from scholarrag.llm.base import ModelTier
from scholarrag.observability.langfuse import observe, update_current_generation

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI

# Test seams mirroring client.chat.completions.create(model=, messages=, ...).
CreateFn = Callable[..., Any]
StreamFn = Callable[..., Iterable[Any]]


class OpenAILLM:
    """OpenAI-compatible :class:`LLMClient`. Resolves tier -> model from settings."""

    def __init__(
        self,
        settings: Settings,
        *,
        create_fn: CreateFn | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        if settings.openai_api_key is None and create_fn is None and stream_fn is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAILLM")
        self._settings = settings
        self._create_fn = create_fn
        self._stream_fn = stream_fn
        self._client: OpenAI | None = None

    def _client_lazy(self) -> OpenAI:
        """Construct (and cache) the OpenAI SDK client on first real use."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,  # None -> OpenAI's default host
            )
        return self._client

    def _model_for_tier(self, tier: ModelTier) -> str:
        if tier == "strong":
            return self._settings.openai_model_strong
        return self._settings.openai_model_cheap

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        """Build the chat ``messages`` list (system first, if given, then the user turn)."""
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _create(self, **kwargs: Any) -> Any:
        """Call chat.completions.create — via the injected stub or the real SDK."""
        if self._create_fn is not None:
            return self._create_fn(**kwargs)
        return self._client_lazy().chat.completions.create(**kwargs)

    @observe(name="openai-complete", as_type="generation")
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "cheap",
        max_tokens: int | None = None,
    ) -> str:
        "Send ``prompt`` to the model and return the text of the reply."
        model = self._model_for_tier(tier)
        response = self._create(
            model=model,
            messages=self._messages(prompt, system),
            max_tokens=max_tokens or self._settings.llm_max_output_tokens,
        )
        text: str = response.choices[0].message.content or ""

        usage = getattr(response, "usage", None)
        update_current_generation(
            model=model,
            input=prompt,
            output=text,
            usage=(
                {
                    "input": usage.prompt_tokens or 0,
                    "output": usage.completion_tokens or 0,
                }
                if usage is not None
                else None
            ),
        )
        return text

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "strong",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream the reply as text deltas (used by the SSE endpoint)."""
        kwargs: dict[str, Any] = {
            "model": self._model_for_tier(tier),
            "messages": self._messages(prompt, system),
            "max_tokens": max_tokens or self._settings.llm_max_output_tokens,
            "stream": True,
        }
        if self._stream_fn is not None:
            chunks: Iterable[Any] = self._stream_fn(**kwargs)
        else:
            chunks = self._client_lazy().chat.completions.create(**kwargs)
        for chunk in chunks:
            if not chunk.choices:
                continue  # some providers send a final usage-only chunk
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
