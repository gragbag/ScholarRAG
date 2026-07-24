"""The agentic pipeline — a LangGraph state machine with bounded self-correction.

One agent, four nodes, two decisions. Instead of the fixed retrieve->generate
script, the graph can judge its own retrieval (``grade``), rewrite the query and
retry when it's weak, and re-try generation when the answer comes back uncited:

    START -> retrieve -> grade --(relevant)--> generate --(cited/refusal)--> END
                  ^        |                      |
                  |     (weak)                (uncited)
                  |        v                      |
                  +-- rewrite_query <-------------+        (loop, capped)

Budget by design: ``grade``/``rewrite`` run on the cheap tier, the answer check
is deterministic (your citation machinery — zero tokens), and ``iterations`` is
a hard cap on the loop. ``enforce_grounding`` stays the last word outside the
graph — the agent gets more chances to succeed, never permission to ship an
ungrounded answer.

``langgraph``/``langchain`` imports stay inside methods (extras, absent in CI).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypedDict

from sqlalchemy.orm import Session

from scholarrag.cache.answer_cache import AnswerCache
from scholarrag.generation.base import Answer
from scholarrag.generation.citations import extract_citations
from scholarrag.generation.prompts import GROUNDED_SYSTEM, format_sources
from scholarrag.guardrails.output import enforce_grounding, looks_like_refusal
from scholarrag.retrieval.base import RetrievedChunk, Retriever

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

_HUMAN_TEMPLATE = "Sources:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"

_GRADER_SYSTEM = (
    "You judge whether retrieved passages could answer a question. "
    "Reply with exactly one word: 'relevant' if the passages plausibly contain "
    "the answer, or 'weak' if they do not."
)

_REWRITE_SYSTEM = (
    "You rewrite search queries for a document retrieval system. The previous "
    "query failed to find relevant passages. Produce ONE alternative search "
    "query — different wording, expanded acronyms, likely synonyms. Output only "
    "the query, nothing else."
)


class AgentState(TypedDict):
    """The shared state every node reads and updates."""

    question: str  # the user's original question — never mutated
    query: str  # the CURRENT retrieval query (rewritten on retries)
    session: Any  # the request's DB session (no checkpointer in Step 1)
    chunks: list[RetrievedChunk]  # latest retrieval results
    verdict: str  # grader's call: "relevant" | "weak"
    answer: str  # latest generation
    iterations: int  # rewrite-loop counter (the safety rail)


class AgenticQueryEngine:
    """LangGraph-driven pipeline: same surface as the others, decisions inside."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: BaseChatModel,  # strong tier — final generation
        decider_llm: BaseChatModel,  # cheap tier — grading + query rewriting
        cache: AnswerCache | None = None,
        top_k: int = 5,
        max_iterations: int = 2,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._decider = decider_llm
        self._cache = cache
        self._top_k = top_k
        self._max_iterations = max_iterations
        self._graph: Any = None  # compiled lazily on first use (exercise A)

    # ── scaffolded nodes (compositions of pieces you've already built) ───────

    def _node_retrieve(self, state: AgentState) -> dict[str, Any]:
        """Hybrid retrieval on the *current* query (which retries may rewrite)."""
        chunks = self._retriever.retrieve(state["session"], state["query"], top_k=self._top_k)
        return {"chunks": chunks}

    def _node_rewrite(self, state: AgentState) -> dict[str, Any]:
        """Cheap-tier query reformulation; spends one unit of the loop budget."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _REWRITE_SYSTEM),
                ("human", "Question: {question}\nFailed query: {query}\nRewritten query:"),
            ]
        )
        chain = prompt | self._decider | StrOutputParser()
        new_query = chain.invoke({"question": state["question"], "query": state["query"]}).strip()
        return {"query": new_query or state["question"], "iterations": state["iterations"] + 1}

    def _node_generate(self, state: AgentState) -> dict[str, Any]:
        """The same LCEL generation chain as the langchain pipeline."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", GROUNDED_SYSTEM), ("human", _HUMAN_TEMPLATE)]
        )
        chain = prompt | self._llm | StrOutputParser()
        text = chain.invoke(
            {"context": format_sources(state["chunks"]), "question": state["question"]}
        )
        return {"answer": text}

    def _node_grade(self, state: AgentState) -> dict[str, Any]:
        "Cheap-tier judgment: can these chunks answer the question?"
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _GRADER_SYSTEM),
                ("human", "Question: {question}\n\nPassages:\n{passages}"),
            ]
        )

        chain = prompt | self._decider | StrOutputParser()

        passages = "\n\n".join(c.text[:300] for c in state["chunks"])

        raw = chain.invoke({"question": state["question"], "passages": passages}).strip().lower()
        if "weak" in raw or "not relevant" in raw:
            verdict = "weak"
        elif "relevant" in raw:
            verdict = "relevant"
        else:
            verdict = "relevant"

        return {"verdict": verdict}

    def _route_after_grade(self, state: AgentState) -> str:
        "Decide: generate from these chunks, or rewrite and retry?"

        if state["verdict"] == "relevant" or state["iterations"] >= self._max_iterations:
            return "generate"

        return "rewrite"

    def _route_after_generate(self, state: AgentState) -> str:
        "Decide: ship this answer, or loop for better evidence?"

        from langgraph.graph import END

        answer = state["answer"]
        if extract_citations(answer) or looks_like_refusal(answer):
            return str(END)

        if state["iterations"] < self._max_iterations:
            return "rewrite"

        return str(END)

    def _build_graph(self) -> Any:
        "Wire nodes + edges into a compiled LangGraph state machine."
        from langgraph.graph import START, StateGraph

        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("grade", self._node_grade)
        graph.add_node("rewrite", self._node_rewrite)
        graph.add_node("generate", self._node_generate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_edge("rewrite", "retrieve")

        graph.add_conditional_edges("grade", self._route_after_grade)
        graph.add_conditional_edges("generate", self._route_after_generate)

        return graph.compile()

    def _graph_lazy(self) -> Any:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ── public surface (scaffolded — mirrors the other engines) ──────────────

    def _run(self, session: Session, question: str) -> AgentState:
        """Invoke the graph from a fresh state."""
        state: AgentState = {
            "question": question,
            "query": question,  # first retrieval uses the question as-is
            "session": session,
            "chunks": [],
            "verdict": "",
            "answer": "",
            "iterations": 0,
        }
        result: AgentState = self._graph_lazy().invoke(state)
        return result

    def _to_answer(self, text: str, chunks: list[RetrievedChunk]) -> Answer:
        """Same exit contract as every pipeline: cite, map, gate."""
        cited = extract_citations(text)
        sources = [chunks[n - 1] for n in cited if 1 <= n <= len(chunks)]
        return enforce_grounding(Answer(text=text, sources=sources))

    def query(self, session: Session, query: str) -> Answer:
        if self._cache is not None:
            hit = self._cache.get(query)
            if hit is not None:
                return hit

        final = self._run(session, query)
        answer = self._to_answer(final["answer"], final["chunks"])
        if self._cache is not None:
            self._cache.put(query, answer)
        return answer

    def answer_with_context(
        self, session: Session, query: str
    ) -> tuple[Answer, list[RetrievedChunk]]:
        final = self._run(session, query)
        return self._to_answer(final["answer"], final["chunks"]), final["chunks"]

    def stream(self, session: Session, query: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Buffer-then-stream: the loop runs to completion, then tokens replay.

        Slower first token than the other pipelines — but uniquely, the agentic
        path's streamed output is FULLY GATED (the answer is checked before any
        token leaves). True per-node event streaming is the Step 2 upgrade.
        """
        final = self._run(session, query)
        answer = self._to_answer(final["answer"], final["chunks"])

        def _replay(text: str) -> Iterator[str]:
            for i in range(0, len(text), 24):
                yield text[i : i + 24]

        # Full candidate list, not answer.sources — the SSE route re-maps [n]
        # citations against this list, and the numbers index the candidates.
        return final["chunks"], _replay(answer.text)
