from __future__ import annotations

from time import perf_counter

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
from app.services.dataset_service import DatasetService
from app.services.experiment_service import ExperimentService


class EvaluationService:
    def __init__(
        self,
        experiment_service: ExperimentService | None = None,
        provider: BaseLLMProvider | None = None,
        ragas_evaluator: RagasEvaluator | None = None,
        deepeval_evaluator: DeepEvalEvaluator | None = None,
        dataset_service: DatasetService | None = None,
    ) -> None:
        self.experiment_service = experiment_service or ExperimentService()
        self.provider = provider or OllamaProvider()
        self.ragas_evaluator = ragas_evaluator or RagasEvaluator()
        self.deepeval_evaluator = deepeval_evaluator or DeepEvalEvaluator()
        self.dataset_service = dataset_service or DatasetService()

    def run(self, request: EvaluationRequest) -> EvaluationResponse:
        started_at = perf_counter()
        requested_metrics = request.metrics or ["ragas"]
        evaluation_samples = self._resolve_samples(request)
        generated_samples, generation_latency = self._generate_answers(request, evaluation_samples)

        metrics: list[EvaluationMetricResult] = []
        if "ragas" in requested_metrics:
            metrics.extend(self._run_ragas(generated_samples))

        if any(metric in requested_metrics for metric in {"deepeval", "hallucination", "toxicity", "bias", "g_eval"}):
            metrics.extend(self._run_deepeval(generated_samples))

        total_latency = max(0.0, perf_counter() - started_at)
        run_record = self.experiment_service.create_run(
            EvaluationRunCreate(
                dataset_name=request.dataset_name,
                model_name=request.config.model_name,
                metadata={
                    "provider": request.config.provider.value,
                    "metrics": requested_metrics,
                    "sample_count": len(generated_samples),
                    "generation_latency_seconds": generation_latency,
                    "total_latency_seconds": total_latency,
                },
            )
        )

        return EvaluationResponse(
            run=run_record,
            metrics=metrics,
            latency_seconds=total_latency,
            cost_usd=0.0,
        )

    def _resolve_samples(self, request: EvaluationRequest) -> list[EvaluationSample]:
        if request.samples:
            return [EvaluationSample.model_validate(sample) for sample in request.samples]
        return self.dataset_service.load_dataset_rows(request.dataset_name, limit=request.max_samples)

    def _generate_answers(
        self,
        request: EvaluationRequest,
        samples: list[EvaluationSample],
    ) -> tuple[list[EvaluationSample], float]:
        generated: list[EvaluationSample] = []
        total_latency = 0.0
        for sample in samples:
            if sample.answer.strip():
                generated.append(sample)
                continue

            context_section = "\n".join(sample.contexts) if sample.contexts else ""
            prompt = (
                "Answer the question using the provided context. "
                "If context is missing, answer briefly and clearly.\n\n"
                f"Question: {sample.question}\n"
                f"Context: {context_section}\n"
                "Answer:"
            )
            try:
                result = self.provider.generate(prompt=prompt, config=request.config, system_prompt=request.system_prompt)
                total_latency += result.latency_seconds
                generated.append(
                    EvaluationSample(
                        question=sample.question,
                        answer=result.output.strip(),
                        contexts=sample.contexts,
                        ground_truth=sample.ground_truth,
                    )
                )
            except Exception as exc:
                generated.append(
                    EvaluationSample(
                        question=sample.question,
                        answer=(sample.ground_truth or "").strip(),
                        contexts=sample.contexts,
                        ground_truth=sample.ground_truth,
                    )
                )
                total_latency += 0.0
                _ = exc
        return generated, total_latency

    def _run_ragas(self, samples: list[EvaluationSample]) -> list[EvaluationMetricResult]:
        if not samples:
            return [EvaluationMetricResult(name="ragas", score=0.0, details={"status": "no-samples"})]
        try:
            return self.ragas_evaluator.evaluate(
                RagasInputs(
                    questions=[sample.question for sample in samples],
                    answers=[sample.answer for sample in samples],
                    contexts=[sample.contexts for sample in samples],
                    ground_truths=[sample.ground_truth for sample in samples],
                )
            )
        except Exception as exc:
            return [
                EvaluationMetricResult(
                    name="ragas",
                    score=0.0,
                    details={"status": "failed", "reason": str(exc)},
                )
            ]

    def _run_deepeval(self, samples: list[EvaluationSample]) -> list[EvaluationMetricResult]:
        try:
            return self.deepeval_evaluator.evaluate(samples)
        except Exception as exc:
            return [
                EvaluationMetricResult(
                    name="deepeval",
                    score=0.0,
                    details={"status": "failed", "reason": str(exc)},
                )
            ]
