"""Langfuse tracing — a thin adapter that is a safe no-op unless configured.

The core pipeline must not depend on Langfuse: CI doesn't install it, tests
must never emit telemetry, and a missing key must never break a query. So this
module wraps the v2 SDK's ``@observe`` decorator behind a call-time gate:

* package not installed  -> decorated functions run undecorated (identity)
* installed, no keys     -> same (``is_enabled()`` is False until configured)
* configured at startup  -> calls flow through Langfuse's real decorator, which
  builds the trace/span/generation tree from the call nesting

``configure_observability(settings)`` is called once at app startup (and by the
manual scripts); everything else here is safe to call from anywhere, any time.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from scholarrag.config import Settings

F = TypeVar("F", bound=Callable[..., Any])

_enabled = False


def is_enabled() -> bool:
    """Whether tracing is active (keys configured and the SDK importable)."""
    return _enabled


def configure_observability(settings: Settings) -> None:
    """Enable Langfuse tracing if keys are configured; otherwise stay a no-op."""
    global _enabled
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return
    try:
        from langfuse.decorators import langfuse_context
    except ImportError:  # observability extra not installed
        return
    langfuse_context.configure(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        enabled=True,
    )
    _enabled = True


def observe(*, name: str | None = None, as_type: str | None = None) -> Callable[[F], F]:
    """Decorator: trace this function as a span (or ``as_type="generation"``).

    Both the raw function and a Langfuse-decorated version are prepared up
    front; each *call* picks one based on ``is_enabled()``, so decoration order
    vs. configuration order doesn't matter and the off-path adds ~zero cost.
    """

    def decorator(fn: F) -> F:
        traced: Callable[..., Any] | None = None
        try:
            from langfuse.decorators import observe as lf_observe

            # v2 quirk: as_type accepts only "generation" or absence — pass
            # name-only for plain spans.
            if as_type is not None:
                traced = lf_observe(name=name, as_type=as_type)(fn)
            else:
                traced = lf_observe(name=name)(fn)
        except ImportError:
            traced = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if traced is not None and _enabled:
                return traced(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def update_current_generation(
    *,
    model: str | None = None,
    input: Any | None = None,
    output: Any | None = None,
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Attach model/usage details to the current generation (no-op when disabled).

    ``usage`` uses Langfuse's generic shape: ``{"input": <prompt_tokens>,
    "output": <completion_tokens>}`` — this is what makes token counts and cost
    appear in the UI.
    """
    if not _enabled:
        return
    from langfuse.decorators import langfuse_context

    langfuse_context.update_current_observation(
        model=model, input=input, output=output, usage=usage, metadata=metadata
    )


def update_current_trace(**kwargs: Any) -> None:
    """Attach metadata/tags/user to the current trace (no-op when disabled)."""
    if not _enabled:
        return
    from langfuse.decorators import langfuse_context

    langfuse_context.update_current_trace(**kwargs)


def flush() -> None:
    """Block until buffered events are sent (call at shutdown / end of scripts)."""
    if not _enabled:
        return
    from langfuse.decorators import langfuse_context

    langfuse_context.flush()


def get_langchain_callbacks(settings: Settings) -> list[Any]:
    """Langfuse's LangChain callback handler (or ``[]`` when tracing is off).

    The LangChain/LangGraph pipelines bypass our instrumented ``GeminiLLM``, so
    their generations would be invisible to Langfuse. Attaching this handler to
    the LangChain chat models restores per-generation tokens/latency there.
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse.callback import CallbackHandler
    except ImportError:  # observability extra not installed
        return []
    return [
        CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    ]
