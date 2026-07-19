"""Query rewriting — expand one user question into several search queries.

Sits *in front of* retrieval. Raw user queries are often bad for search
(vocabulary mismatch, acronyms, under-specification); rewriting uses the *cheap*
LLM tier to produce alternative phrasings. Step 4 will run retrieval for each
variation and fuse the results with :func:`reciprocal_rank_fusion` — multi-query
retrieval, reusing the RRF you built in Step 2.

Two pieces:
* :func:`parse_query_list` — turn messy LLM text into a clean list of queries.
* :class:`QueryRewriter`   — call the LLM and return original + variations.
"""

from __future__ import annotations

import re

from scholarrag.llm.base import LLMClient
from scholarrag.llm.prompts import QUERY_REWRITE_SYSTEM, render_query_rewrite_prompt


def parse_query_list(text: str) -> list[str]:
    "Parse an LLM's line-per-query response into a clean, deduped list."

    result = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()

        if cleaned and cleaned not in result:
            result.append(cleaned)

    return result


class QueryRewriter:
    """Expands a query into several search variations using the cheap LLM tier."""

    def __init__(self, *, llm: LLMClient, num_variations: int = 3) -> None:
        self._llm = llm
        self._num_variations = num_variations

    def rewrite(self, query: str) -> list[str]:
        "Return the original query plus LLM-generated variations (deduped)."

        prompt = render_query_rewrite_prompt(query, self._num_variations)
        raw = self._llm.complete(prompt, system=QUERY_REWRITE_SYSTEM, tier="cheap")

        variations = parse_query_list(raw)
        result = [query]
        for v in variations:
            if v not in result:
                result.append(v)

        return result
