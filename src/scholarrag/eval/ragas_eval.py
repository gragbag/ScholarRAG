"""Generation evaluation with RAGAS (LLM-as-judge), driven through LangChain.

Step 2 measures *answer* quality, not just retrieval: faithfulness (no
hallucination), answer relevancy, context precision, context recall. RAGAS calls
its judge through LangChain's model/embeddings interfaces, so you hand it a
LangChain-wrapped Gemini (judge LLM) and BGE (judge embeddings) — your first
taste of LangChain as the provider-agnostic adapter layer.

``ragas`` / ``langchain`` are imported lazily inside the exercise functions (they
live in the ``eval`` extra, absent in CI), so this module imports fine without
them and ``collect_samples`` stays testable with the fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from scholarrag.config import Settings
from scholarrag.eval.dataset import EvalExample
from scholarrag.pipeline import QueryEngine


@dataclass(frozen=True, slots=True)
class GenerationSample:
    """One row of the generation-eval dataset: what RAGAS scores."""

    question: str
    answer: str
    contexts: list[str]  # the retrieved passages the answer was generated from
    reference: str | None  # ground-truth answer (for context recall/precision)


def collect_samples(
    engine: QueryEngine, session: Session, examples: list[EvalExample]
) -> list[GenerationSample]:
    """Run the full pipeline over each example, capturing answer + contexts."""
    samples: list[GenerationSample] = []
    for example in examples:
        answer, chunks = engine.answer_with_context(session, example.question)
        samples.append(
            GenerationSample(
                question=example.question,
                answer=answer.text,
                contexts=[chunk.text for chunk in chunks],
                reference=example.reference_answer,
            )
        )
    return samples


def build_judge(settings: Settings) -> tuple[Any, Any]:
    "Build the RAGAS judge: a LangChain-wrapped Gemini LLM + BGE embeddings."

    from langchain_google_genai import ChatGoogleGenerativeAI

    chat = ChatGoogleGenerativeAI(
        model=settings.gemini_model_cheap,
        google_api_key=settings.gemini_api_key,
    )

    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def run_ragas_eval(
    samples: list[GenerationSample],
    *,
    llm: Any,
    embeddings: Any,
    max_workers: int = 2,
) -> dict[str, float]:
    "Score ``samples`` with the four RAGAS metrics; return mean per metric."
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import SingleTurnSample

    rows = [
        SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=s.contexts,
            reference=s.reference or "",
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=rows)

    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=max_workers),
    )

    df = result.to_pandas()
    return {m.name: float(df[m.name].mean()) for m in metrics}
