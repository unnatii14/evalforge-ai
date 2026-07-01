from __future__ import annotations

from time import perf_counter

from app.evaluators.deepeval_evaluator import DeepEvalEvaluator
from app.evaluators.ragas_evaluator import RagasEvaluator, RagasInputs
from app.llm_providers.base import BaseLLMProvider
from app.llm_providers.factory import get_provider
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationRunCreate,
    EvaluationSample,
    EvaluationSampleRecord,
    LLMConfig,
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
        # When no provider is injected (e.g. in tests), the provider is resolved
        # per-run from the request config via the factory, so the requested
        # provider actually routes instead of always using Ollama.
        self._provider_override = provider
        self.ragas_evaluator = ragas_evaluator or RagasEvaluator()
        self.deepeval_evaluator = deepeval_evaluator or DeepEvalEvaluator()
        self.dataset_service = dataset_service or DatasetService()

    def _resolve_provider(self, config: LLMConfig) -> BaseLLMProvider:
        return self._provider_override or get_provider(config)

    def run(self, request: EvaluationRequest) -> EvaluationResponse:
        started_at = perf_counter()
        requested_metrics = request.metrics or ["ragas"]
        evaluation_samples = self._resolve_samples(request)
        generated_samples, sources, generation_latency = self._generate_answers(request, evaluation_samples)

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

        self.experiment_service.save_metrics(run_record.id, metrics)
        self.experiment_service.save_samples(
            run_record.id,
            [
                EvaluationSampleRecord(
                    question=sample.question,
                    answer=sample.answer,
                    contexts=sample.contexts,
                    ground_truth=sample.ground_truth,
                    answer_source=source,
                )
                for sample, source in zip(generated_samples, sources)
            ],
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
    ) -> tuple[list[EvaluationSample], list[str], float]:
        generated: list[EvaluationSample] = []
        sources: list[str] = []
        total_latency = 0.0
        provider = self._resolve_provider(request.config)
        for sample in samples:
            if sample.answer.strip():
                generated.append(sample)
                sources.append("provided")
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
                result = provider.generate(prompt=prompt, config=request.config, system_prompt=request.system_prompt)
                total_latency += result.latency_seconds
                generated.append(
                    EvaluationSample(
                        question=sample.question,
                        answer=result.output.strip(),
                        contexts=sample.contexts,
                        ground_truth=sample.ground_truth,
                    )
                )
                sources.append("generated")
            except Exception:
                # Generation failed. Do NOT fall back to the ground truth as the
                # answer -- that would make a failed run score near-perfect and
                # corrupt the evaluation. Record an empty answer so the metrics
                # reflect the failure honestly.
                generated.append(
                    EvaluationSample(
                        question=sample.question,
                        answer="",
                        contexts=sample.contexts,
                        ground_truth=sample.ground_truth,
                    )
                )
                sources.append("generation_failed")
        return generated, sources, total_latency

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
