from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.core.config import get_settings
from app.evaluators.base import BaseEvaluator
from app.models.schemas import EvaluationMetricResult


@dataclass(frozen=True)
class RagasInputs:
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str]


class RagasEvaluator(BaseEvaluator):
    """RAGAS quality metrics wired to a local Ollama judge or a hosted Groq judge.

    RAGAS defaults to OpenAI for its judge LLM and embeddings, which would require
    an OPENAI_API_KEY. This evaluator injects the backend selected by
    ``settings.eval_backend``:

    * ``ollama`` (default) -> local Ollama chat + embedding models (fully offline).
    * ``groq``            -> Groq's free hosted chat model as the judge, plus a
                             small local Hugging Face embedding model (Groq has no
                             embeddings endpoint). Used for cloud deployments.
    """

    def __init__(self, judge_model: str | None = None, embedding_model: str | None = None) -> None:
        settings = get_settings()
        self.backend = settings.eval_backend
        self.judge_model = judge_model or settings.judge_model
        self.embedding_model = embedding_model or settings.embedding_model
        self.ollama_host = settings.ollama_host
        self.groq_api_key = settings.groq_api_key
        self.groq_model = settings.groq_model
        self.groq_base_url = settings.groq_base_url
        self.hf_embedding_model = settings.hf_embedding_model

    # ---- LLM builders -------------------------------------------------
    def _build_llm(self):
        from ragas.llms import LangchainLLMWrapper

        if self.backend == "groq":
            from langchain_openai import ChatOpenAI

            return LangchainLLMWrapper(
                ChatOpenAI(
                    model=self.groq_model,
                    api_key=self.groq_api_key,
                    base_url=self.groq_base_url,
                    temperature=0.0,
                )
            )

        from langchain_community.chat_models import ChatOllama

        chat_kwargs = {"model": self.judge_model, "temperature": 0.0}
        if self.ollama_host:
            chat_kwargs["base_url"] = self.ollama_host
        return LangchainLLMWrapper(ChatOllama(**chat_kwargs))

    def _build_embeddings(self):
        from ragas.embeddings import LangchainEmbeddingsWrapper

        if self.backend == "groq":
            # Groq has no embeddings API -> use a small local sentence-transformers model.
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=self.hf_embedding_model))

        from langchain_community.embeddings import OllamaEmbeddings

        embed_kwargs = {"model": self.embedding_model}
        if self.ollama_host:
            embed_kwargs["base_url"] = self.ollama_host
        return LangchainEmbeddingsWrapper(OllamaEmbeddings(**embed_kwargs))

    # ---- Evaluation ---------------------------------------------------
    def evaluate(self, inputs: RagasInputs) -> list[EvaluationMetricResult]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise RuntimeError("RAGAS is not available in the current environment.") from exc

        dataset = Dataset.from_pandas(
            pd.DataFrame(
                {
                    "question": inputs.questions,
                    "answer": inputs.answers,
                    "contexts": inputs.contexts,
                    "ground_truth": inputs.ground_truths,
                }
            )
        )

        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness],
            llm=self._build_llm(),
            embeddings=self._build_embeddings(),
        )
        results_frame = results.to_pandas()
        metric_columns = [
            column
            for column in results_frame.columns
            if column not in {"question", "answer", "contexts", "ground_truth"}
        ]
        judge = self.groq_model if self.backend == "groq" else self.judge_model
        metrics: list[EvaluationMetricResult] = []
        for column in metric_columns:
            series = results_frame[column].dropna()
            if series.empty:
                continue
            score = max(0.0, min(1.0, float(series.mean())))
            metrics.append(
                EvaluationMetricResult(
                    name=column,
                    score=score,
                    details={"rows": len(series), "backend": self.backend, "judge_model": judge},
                )
            )
        return metrics
