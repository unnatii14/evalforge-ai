from __future__ import annotations

from app.evaluators.deepeval_evaluator import DeepEvalEvaluator
from app.evaluators.ragas_evaluator import RagasEvaluator, RagasInputs
from app.llm_providers.base import BaseLLMProvider
from app.llm_providers.ollama_provider import OllamaProvider
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationRunCreate,
    EvaluationSample,
)
from app.services.experiment_service import ExperimentService


class EvaluationService:
    def __init__(
        self,
        experiment_service: ExperimentService | None = None,
        provider: BaseLLMProvider | None = None,
        ragas_evaluator: RagasEvaluator | None = None,
        deepeval_evaluator: DeepEvalEvaluator | None = None,
    ) -> None:
        self.experiment_service = experiment_service or ExperimentService()
        self.provider = provider or OllamaProvider()
        self.ragas_evaluator = ragas_evaluator or RagasEvaluator()
        self.deepeval_evaluator = deepeval_evaluator or DeepEvalEvaluator()

    def run(self, request: EvaluationRequest) -> EvaluationResponse:
        run_record = self.experiment_service.create_run(
            EvaluationRunCreate(
                dataset_name=request.dataset_name,
                model_name=request.config.model_name,
                metadata={"provider": request.config.provider.value, "metrics": request.metrics},
            )
        )

        metrics: list[EvaluationMetricResult] = []
        evaluation_samples = [EvaluationSample.model_validate(sample) for sample in request.samples]
        if "ragas" in request.metrics:
            if evaluation_samples:
                metrics.extend(
                    self.ragas_evaluator.evaluate(
                        RagasInputs(
                            questions=[sample.question for sample in evaluation_samples],
                            answers=[sample.answer for sample in evaluation_samples],
                            contexts=[sample.contexts for sample in evaluation_samples],
                            ground_truths=[sample.ground_truth for sample in evaluation_samples],
                        )
                    )
                )
            else:
                metrics.append(
                    EvaluationMetricResult(
                        name="ragas",
                        score=0.0,
                        details={"status": "pending-samples"},
                    )
                )
        if any(metric in request.metrics for metric in {"deepeval", "hallucination", "toxicity", "bias", "g_eval"}):
            metrics.extend(self.deepeval_evaluator.evaluate())

        return EvaluationResponse(
            run=run_record,
            metrics=metrics,
            latency_seconds=0.0,
            cost_usd=0.0,
        )
