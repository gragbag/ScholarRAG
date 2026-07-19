"""LLM client tests.

``FakeLLM`` and the protocol test pass now. The Anthropic test is Exercise A —
remove its ``@pytest.mark.skip`` once you've implemented ``AnthropicLLM.complete``
(it injects a stub ``create_fn``, so no SDK and no API key).
"""

from __future__ import annotations

from dataclasses import dataclass

from scholarrag.config import Settings
from scholarrag.llm import AnthropicLLM, FakeLLM, LLMClient


def _settings() -> Settings:
    return Settings(anthropic_api_key=None, llm_model_cheap="haiku-x", llm_model_strong="sonnet-x")


def test_clients_conform_to_protocol() -> None:
    assert isinstance(FakeLLM(), LLMClient)
    assert isinstance(AnthropicLLM(_settings(), create_fn=lambda **kw: None), LLMClient)


def test_fake_llm_scripts_and_records_calls() -> None:
    llm = FakeLLM(["first", "second"], default="fallback")
    assert llm.complete("a", tier="cheap") == "first"
    assert llm.complete("b", system="sys", tier="strong") == "second"
    assert llm.complete("c") == "fallback"  # scripted responses exhausted
    assert llm.calls[1] == {"prompt": "b", "system": "sys", "tier": "strong"}


# ── Exercise A — AnthropicLLM.complete ───────────────────────────────────────
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
