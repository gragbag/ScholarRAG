"""Deterministic, dependency-free LLM client for tests and CI.

No SDK, no API key, no network — the LLM counterpart to ``FakeEmbedder`` and
``LocalVectorStore``. Either returns a fixed string or replays a scripted list of
responses, and records every call so tests can assert on what was sent.
"""

from __future__ import annotations

from collections.abc import Iterator

from scholarrag.llm.base import ModelTier


class FakeLLM:
    """A scriptable :class:`LLMClient` implementation for tests.

    ``responses`` (if given) are returned in order, one per ``complete`` call;
    once exhausted (or if omitted) it returns ``default``. Each call is recorded
    in ``self.calls`` as a dict, so a test can assert the tier/system/prompt used.
    """

    def __init__(self, responses: list[str] | None = None, *, default: str = "ok") -> None:
        self._responses = list(responses) if responses else []
        self._default = default
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "cheap",
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system, "tier": tier})
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "strong",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield the next response in fixed-size slices, mimicking token streaming."""
        self.calls.append({"prompt": prompt, "system": system, "tier": tier})
        text = self._responses.pop(0) if self._responses else self._default
        for i in range(0, len(text), 8):
            yield text[i : i + 8]
