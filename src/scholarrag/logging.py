"""Structured logging with per-request correlation IDs.

Uses ``structlog`` so every log line is a JSON object (in production/CI) carrying
a ``correlation_id`` bound for the lifetime of a request. The correlation id is
stored in a :class:`contextvars.ContextVar` so it propagates across ``await``
boundaries without being threaded through every call.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import cast

import structlog
from structlog.types import EventDict, WrappedLogger

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(value: str | None) -> None:
    """Bind ``value`` as the correlation id for the current context."""
    _correlation_id.set(value)


def get_correlation_id() -> str | None:
    """Return the correlation id bound to the current context, if any."""
    return _correlation_id.get()


def _add_correlation_id(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict["correlation_id"] = cid
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure ``structlog`` and the stdlib logging bridge.

    Idempotent: safe to call more than once (e.g. app startup + tests).
    """
    logging.basicConfig(format="%(message)s", level=level.upper())

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_correlation_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
