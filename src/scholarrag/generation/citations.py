"""Citation parsing — pull the ``[n]`` markers back out of the answer text.

Same LLM-output-wrangling idea as ``parse_query_list``: the model writes ``[1]``,
``[2]`` inline, and we recover which sources it actually cited so we can return
just those (not every retrieved chunk).
"""

from __future__ import annotations

import re


def extract_citations(text: str) -> list[int]:
    "Return the unique citation numbers in ``text``, in order of first appearance."

    matches = re.findall(r"\[(\d+)\]", text)
    result: list[int] = []
    for m in matches:
        n = int(m)
        if n not in result:
            result.append(n)

    return result
