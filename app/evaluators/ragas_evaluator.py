from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.evaluators.base import BaseEvaluator
from app.models.schemas import EvaluationMetricResult


@dataclass(frozen=True)
class RagasInputs:
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str]


class RagasEvaluator(BaseEvaluator):
    def evaluate(self, inputs: RagasInputs) -> list[EvaluationMetricResult]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness
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
        )
        results_frame = results.to_pandas()
        metric_columns = [column for column in results_frame.columns if column not in {"question", "answer", "contexts", "ground_truth"}]
        return [
            EvaluationMetricResult(name=column, score=float(results_frame[column].mean()), details={"rows": len(results_frame)})
            for column in metric_columns
        ]
