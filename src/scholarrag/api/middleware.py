"""ASGI middleware: attach a correlation id to every request."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

from scholarrag.logging import set_correlation_id

CORRELATION_HEADER = "X-Correlation-ID"

RequestHandler = Callable[[Request], Awaitable[Response]]


async def correlation_id_middleware(request: Request, call_next: RequestHandler) -> Response:
    """Bind an inbound (or freshly generated) correlation id for the request.

    Honours an incoming ``X-Correlation-ID`` header so ids can be traced across
    service boundaries, and echoes the id back on the response.
    """
    correlation_id = request.headers.get(CORRELATION_HEADER) or uuid.uuid4().hex
    set_correlation_id(correlation_id)
    try:
        response = await call_next(request)
    finally:
        set_correlation_id(None)
    response.headers[CORRELATION_HEADER] = correlation_id
    return response
