"""Streamlit chat UI for ScholarRAG — run with ``make ui`` (backend must be up).

A thin HTTP client over the FastAPI backend: it streams answers from
``/query/stream`` and renders the cited sources. No pipeline, no models here —
everything runs behind the API. Because Streamlit makes the HTTP call
server-side (its own Python process, not the browser), there is no CORS to
configure; the backend is untouched.

Launch:  streamlit run src/scholarrag/ui/app.py   (or: make ui)

This module is a Streamlit entry-point script, never imported by the package or
the tests — the testable logic all lives in ``client.py``.
"""

from __future__ import annotations

import streamlit as st

from scholarrag.ui.client import (
    BackendUnavailable,
    RateLimited,
    ScholarRAGClient,
    Source,
    StreamSink,
    api_url,
)


@st.cache_resource  # type: ignore[untyped-decorator]  # streamlit decorator is untyped
def get_client() -> ScholarRAGClient:
    """One HTTP client per Streamlit server process (cached across script reruns)."""
    return ScholarRAGClient()


def render_sources(sources: list[Source]) -> None:
    """Show the cited chunks in a collapsible panel — the 'grounded' payoff."""
    with st.expander(f"📚 {len(sources)} cited source(s)"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {src.filename}** · chunk {src.chunk_index}")
            st.caption(src.text)


def _replay_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])


def _answer(prompt: str) -> None:
    """Stream one assistant turn and append it to the history."""
    with st.chat_message("assistant"):
        sink = StreamSink()
        try:
            text = st.write_stream(get_client().stream(prompt, sink))
        except RateLimited as exc:
            wait = f" (retry in {exc.retry_after}s)" if exc.retry_after else ""
            st.warning(f"⏳ {exc}{wait}")
            return
        except BackendUnavailable:
            st.error("⚠️ Can't reach the backend. Start it with `make run`, then retry.")
            return

        if sink.ungrounded:
            st.info("The answer wasn't grounded in the corpus, so no sources are cited.")
        elif sink.sources:
            render_sources(sink.sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": text, "sources": sink.sources}
        )


def main() -> None:
    st.set_page_config(page_title="ScholarRAG", page_icon="📖", layout="centered")
    st.title("📖 ScholarRAG")
    st.caption(f"Grounded, cited answers over your corpus · backend: {api_url()}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _replay_history()

    prompt = st.chat_input("Ask a question about the papers…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    _answer(prompt)


main()
