"""Ollama LLM client — free, unlimited, fully-local generation.

Same :class:`LLMClient` protocol as the Claude/Gemini clients, so switching is a
one-line config change (``LLM_PROVIDER=ollama``). Unlike those there's no vendor
SDK and no API key: Ollama is a local HTTP server (``ollama serve`` on :11434)
that you talk to with plain JSON over ``httpx`` (a core dep). Pull a model once
(``ollama pull llama3.1:8b``) and generation is free and offline forever — no
rate limits, which is exactly why it's the nicest backend for the dev loop.

``post_fn`` / ``stream_fn`` are test seams (mirroring Gemini's ``generate_fn`` /
``stream_fn``): inject them to exercise the request/parse logic without a running
Ollama server, a model, or a network call.

Ollama's chat API — ``POST {base_url}/api/chat``::

    request  {"model": "...",
              "messages": [{"role": "system"|"user", "content": "..."}, ...],
              "stream": bool,
              "options": {"num_predict": <max_tokens>}}
    response {"message": {"role": "assistant", "content": "..."}, "done": true, ...}

When ``"stream": true`` the body is **newline-delimited JSON**: one object per
line, each carrying a partial ``message.content`` delta, until an object with
``"done": true``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from scholarrag.config import Settings
from scholarrag.llm.base import ModelTier
from scholarrag.observability.langfuse import (
    observe,
    update_current_generation,
)

if TYPE_CHECKING:  # pragma: no cover
    import httpx

# Test seams. Both are called with (path=..., json=<payload dict>).
#   post_fn   -> the parsed JSON response dict (what a real Response.json() returns)
#   stream_fn -> an iterable of raw NDJSON lines (str)
PostFn = Callable[..., dict[str, Any]]
StreamFn = Callable[..., Iterable[str]]

_CHAT_PATH = "/api/chat"


class OllamaLLM:
    """Ollama-backed :class:`LLMClient`. Resolves tier -> model from settings.

    Construction does no I/O: the httpx client is built lazily on first real call
    (and never at all when a test seam is injected).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        post_fn: PostFn | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        self._settings = settings
        self._post_fn = post_fn
        self._stream_fn = stream_fn
        self._client: httpx.Client | None = None

    # ── plumbing (done — use these from complete/stream) ─────────────────────
    def _client_lazy(self) -> httpx.Client:
        """Construct (and cache) the httpx client pointed at the Ollama server."""
        if self._client is None:
            import httpx

            # Generation on a CPU can take a while, so the timeout is generous.
            self._client = httpx.Client(base_url=self._settings.ollama_base_url, timeout=120.0)
        return self._client

    def _model_for_tier(self, tier: ModelTier) -> str:
        if tier == "strong":
            return self._settings.ollama_model_strong
        return self._settings.ollama_model_cheap

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        """Build the chat ``messages`` list (system first, if given, then the user turn)."""
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _options(self, max_tokens: int | None) -> dict[str, int]:
        """Ollama's per-request knobs. ``num_predict`` is its name for the token cap."""
        return {"num_predict": max_tokens or self._settings.llm_max_output_tokens}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a non-streaming chat request; return the parsed JSON dict."""
        if self._post_fn is not None:
            return self._post_fn(path=_CHAT_PATH, json=payload)
        response = self._client_lazy().post(_CHAT_PATH, json=payload)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def _iter_lines(self, payload: dict[str, Any]) -> Iterator[str]:
        """Yield the raw NDJSON lines of a streaming chat request."""
        if self._stream_fn is not None:
            yield from self._stream_fn(path=_CHAT_PATH, json=payload)
            return
        with self._client_lazy().stream("POST", _CHAT_PATH, json=payload) as response:
            response.raise_for_status()
            yield from response.iter_lines()

    @observe(name="ollama-complete", as_type="generation")
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "cheap",
        max_tokens: int | None = None,
    ) -> str:
        "Send ``prompt`` to ``/api/chat`` (non-streaming) and return the reply text."
        model = self._model_for_tier(tier)
        payload = {
            "model": model,
            "messages": self._messages(prompt, system),
            "stream": False,
            "options": self._options(max_tokens),
        }

        data = self._post(payload)
        text: str = data["message"]["content"]
        update_current_generation(model=model, input=prompt, output=text)

        return text

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "strong",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        "Stream Ollama's reply as text deltas (used by the SSE endpoint)."

        model = self._model_for_tier(tier)
        payload = {
            "model": model,
            "messages": self._messages(prompt, system),
            "stream": True,
            "options": self._options(max_tokens),
        }

        for line in self._iter_lines(payload):
            if not line:
                continue

            chunk = json.loads(line)
            delta = chunk.get("message", {}).get("content")
            if delta:
                yield delta

            if chunk.get("done"):
                break
