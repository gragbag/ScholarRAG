"""LLM client tests.

Covers every backend behind the ``LLMClient`` protocol — ``FakeLLM`` (tests/CI),
``AnthropicLLM`` (Claude), and ``GeminiLLM`` (Google) — none preferred over the
others. Each real client is exercised through an injected seam
(``create_fn`` / ``generate_fn`` / ``stream_fn``), so no test needs a real SDK,
an API key, or a network call.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholarrag.config import Settings
from scholarrag.llm import AnthropicLLM, FakeLLM, GeminiLLM, LLMClient


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # ignore the developer's local .env
        anthropic_api_key=None,
        gemini_api_key=None,
        llm_model_cheap="haiku-x",
        llm_model_strong="sonnet-x",
        gemini_model_cheap="flash-lite-x",
        gemini_model_strong="flash-x",
    )


def test_clients_conform_to_protocol() -> None:
    assert isinstance(FakeLLM(), LLMClient)
    assert isinstance(AnthropicLLM(_settings(), create_fn=lambda **kw: None), LLMClient)
    assert isinstance(GeminiLLM(_settings(), generate_fn=lambda **kw: None), LLMClient)


def test_fake_llm_scripts_and_records_calls() -> None:
    llm = FakeLLM(["first", "second"], default="fallback")
    assert llm.complete("a", tier="cheap") == "first"
    assert llm.complete("b", system="sys", tier="strong") == "second"
    assert llm.complete("c") == "fallback"  # scripted responses exhausted
    assert llm.calls[1] == {"prompt": "b", "system": "sys", "tier": "strong"}


def test_fake_llm_stream_reassembles_to_full_text() -> None:
    llm = FakeLLM(["a grounded answer that streams in pieces"])
    chunks = list(llm.stream("q", tier="strong"))
    assert len(chunks) > 1  # actually chunked
    assert "".join(chunks) == "a grounded answer that streams in pieces"


def test_anthropic_stream_uses_injected_stream_fn() -> None:
    seen: dict[str, object] = {}

    def fake_stream(**kwargs: object) -> list[str]:
        seen.update(kwargs)
        return ["hel", "lo ", "world"]

    llm = AnthropicLLM(_settings(), stream_fn=fake_stream)
    assert "".join(llm.stream("Q", tier="strong")) == "hello world"
    assert seen["model"] == "sonnet-x"
    assert seen["thinking"] == {"type": "disabled"}


# ── AnthropicLLM.complete ────────────────────────────────────────────────────
# Minimal stubs mimicking the Anthropic SDK's response shape.
@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Response:
    content: list[_Block]


def test_anthropic_complete_maps_tier_and_extracts_text() -> None:
    seen: dict[str, object] = {}

    def fake_create(**kwargs: object) -> _Response:
        seen.update(kwargs)
        # Two text blocks + a non-text block that must be ignored.
        return _Response(
            [_Block("text", "hello "), _Block("thinking", "X"), _Block("text", "world")]
        )

    llm = AnthropicLLM(_settings(), create_fn=fake_create)
    out = llm.complete("Q", system="be brief", tier="strong", max_tokens=42)

    assert out == "hello world"  # only text blocks, concatenated
    assert seen["model"] == "sonnet-x"  # strong tier -> strong model
    assert seen["max_tokens"] == 42
    assert seen["system"] == "be brief"
    assert seen["messages"] == [{"role": "user", "content": "Q"}]
    assert "temperature" not in seen  # Sonnet 5 would 400 on it


# ── Gemini provider ──────────────────────────────────────────────────────────
@dataclass
class _GeminiResponse:
    text: str | None


@dataclass
class _GeminiChunk:
    text: str | None


def test_gemini_stream_uses_injected_stream_fn() -> None:
    seen: dict[str, object] = {}

    def fake_stream(**kwargs: object) -> list[_GeminiChunk]:
        seen.update(kwargs)
        # A None-text chunk must be skipped, not yielded.
        return [_GeminiChunk("hel"), _GeminiChunk(None), _GeminiChunk("lo")]

    llm = GeminiLLM(_settings(), stream_fn=fake_stream)
    assert "".join(llm.stream("Q", tier="cheap")) == "hello"
    assert seen["model"] == "flash-lite-x"  # cheap tier -> cheap model
    assert seen["contents"] == "Q"
    assert seen["config"]["thinking_config"] == {"thinking_budget": 0}  # type: ignore[index]


# ── GeminiLLM.complete ───────────────────────────────────────────────────────
def test_gemini_complete_maps_tier_and_extracts_text() -> None:
    seen: dict[str, object] = {}

    def fake_generate(**kwargs: object) -> _GeminiResponse:
        seen.update(kwargs)
        return _GeminiResponse("grounded gemini answer")

    llm = GeminiLLM(_settings(), generate_fn=fake_generate)
    out = llm.complete("Q", system="be brief", tier="strong", max_tokens=42)

    assert out == "grounded gemini answer"
    assert seen["model"] == "flash-x"  # strong tier -> strong model
    assert seen["contents"] == "Q"
    assert seen["config"]["max_output_tokens"] == 42  # type: ignore[index]
    assert seen["config"]["system_instruction"] == "be brief"  # type: ignore[index]
