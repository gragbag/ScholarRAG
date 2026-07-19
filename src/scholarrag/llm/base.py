"""LLMClient protocol and shared types.

An LLM client turns a prompt into text, behind a swappable interface — exactly
like :class:`~scholarrag.embeddings.Embedder` and
:class:`~scholarrag.vectorstore.VectorStore`. Feature code depends on this
protocol, never on a concrete SDK, so Claude → Gemini → a local Ollama model is
a one-line config change and tests run against a :class:`FakeLLM`.

The key idea is the **semantic tier**. Callers ask for ``"cheap"`` or
``"strong"`` thinking, not a model name; the client maps that to the actual
model from settings (``LLM_MODEL_CHEAP`` / ``LLM_MODEL_STRONG``). Query rewriting
uses ``cheap`` (Haiku); Step 4's grounded generation will use ``strong`` (Sonnet).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

# Which quality/cost tier a call needs. The client resolves this to a model id.
ModelTier = Literal["cheap", "strong"]


class LLMError(Exception):
    """Raised when an LLM call fails in a way the caller should handle."""


@runtime_checkable
class LLMClient(Protocol):
    """Text-in, text-out. Constructing one must not do heavy I/O or network."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: ModelTier = "cheap",
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text response to ``prompt``.

        (No ``temperature`` — current Claude models like Sonnet 5 reject sampling
        parameters with a 400; behaviour is steered by the prompt instead.)
        """
        ...
