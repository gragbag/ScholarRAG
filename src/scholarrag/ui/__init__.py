"""ScholarRAG UI — a thin HTTP client + Streamlit chat app over the backend.

Only the client layer is exported here; ``app.py`` is a Streamlit entry-point
script (run via ``streamlit run``), never imported, so importing this package
never pulls in the optional ``streamlit`` dependency.
"""

from scholarrag.ui.client import (
    Answer,
    BackendUnavailable,
    RateLimited,
    ScholarRAGClient,
    Source,
    SSEEvent,
    StreamSink,
    api_url,
    iter_answer_tokens,
    parse_sse_stream,
    stream_answer,
)

__all__ = [
    "Answer",
    "BackendUnavailable",
    "RateLimited",
    "SSEEvent",
    "ScholarRAGClient",
    "Source",
    "StreamSink",
    "api_url",
    "iter_answer_tokens",
    "parse_sse_stream",
    "stream_answer",
]
