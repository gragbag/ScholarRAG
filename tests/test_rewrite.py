"""Query rewriting tests (Exercises B & C — hermetic, use FakeLLM).

Remove each ``@pytest.mark.skip`` once you've implemented the target function.
"""

from __future__ import annotations

from scholarrag.llm import FakeLLM
from scholarrag.retrieval.rewrite import QueryRewriter, parse_query_list


# ── Exercise C — parse_query_list ────────────────────────────────────────────
def test_parse_query_list_cleans_and_dedupes() -> None:
    raw = (
        "1. what is RAG\n\n- retrieval augmented generation\n2) what is RAG\n* how does RAG work\n"
    )
    assert parse_query_list(raw) == [
        "what is RAG",
        "retrieval augmented generation",
        "how does RAG work",  # "what is RAG" appears twice -> deduped
    ]


# ── Exercise B — QueryRewriter.rewrite ───────────────────────────────────────
def test_query_rewriter_includes_original_and_variations() -> None:
    llm = FakeLLM(["neural nets basics\ndeep learning intro\nhow do neural networks work"])
    rewriter = QueryRewriter(llm=llm, num_variations=3)

    result = rewriter.rewrite("how do neural networks work")

    assert result[0] == "how do neural networks work"  # original always first
    assert "neural nets basics" in result
    assert "deep learning intro" in result
    # The variation duplicating the original is not repeated.
    assert result.count("how do neural networks work") == 1
    # It asked the cheap tier.
    assert llm.calls[0]["tier"] == "cheap"
