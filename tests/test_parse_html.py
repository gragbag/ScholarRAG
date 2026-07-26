"""HTML readable-text extraction (Phase 8, Step 2, exercise 1).

Needs the ``html`` extra (trafilatura) — skips in core-only CI.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trafilatura")  # the `html` extra

_PAGE = b"""
<html><body>
  <nav>Home About Contact Login Signup Newsletter</nav>
  <article>
    <h1>Understanding Retrieval-Augmented Generation</h1>
    <p>Retrieval-augmented generation combines a retriever with a generator to
       ground answers in source documents. The retriever fetches relevant
       passages and the generator conditions its output on them, which sharply
       reduces hallucination and improves factual accuracy on knowledge tasks.</p>
    <p>Because the sources are explicit, answers can cite exactly where each claim
       came from, and the corpus can be updated without retraining the model.</p>
  </article>
  <footer>Copyright 2026 BuyNow Ads Ads Sponsored Content</footer>
</body></html>
"""


def test_extract_html_strips_boilerplate() -> None:
    from scholarrag.ingestion.parse import extract_text

    text = extract_text(_PAGE, "html")
    assert "Retrieval-augmented generation" in text  # the article survives
    assert "Home About Contact" not in text  # nav stripped
    assert "Sponsored Content" not in text  # footer/ads stripped
