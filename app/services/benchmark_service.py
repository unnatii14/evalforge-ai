from __future__ import annotations

from statistics import mean

from app.models.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkRow,
    EvaluationRequest,
)
from app.services.evaluation_service import EvaluationService
from app.services.experiment_service import ExperimentService


class BenchmarkService:
    def __init__(
        self,
        evaluation_service: EvaluationService | None = None,
        experiment_service: ExperimentService | None = None,
    ) -> None:
        self.evaluation_service = evaluation_service or EvaluationService()
        self.experiment_service = experiment_service or ExperimentService()

    def run(self, request: BenchmarkRequest) -> BenchmarkResponse:
        rows: list[BenchmarkRow] = []

        for scenario in request.scenarios:
            evaluation_response = self.evaluation_service.run(
                EvaluationRequest(
                    dataset_name=request.benchmark_name,
                    config=scenario.config,
                    metrics=scenario.metrics,
                    samples=scenario.samples,
                )
            )

            metric_scores = evaluation_response.metrics
            overall_score = mean([metric.score for metric in metric_scores]) if metric_scores else 0.0

            rows.append(
                BenchmarkRow(
                    scenario_name=scenario.name,
                    model_name=evaluation_response.run.model_name,
                    provider=scenario.config.provider,
                    overall_score=overall_score,
                    latency_seconds=evaluation_response.latency_seconds,
                    metric_scores=metric_scores,
                )
            )

        best_scenario_name = max(rows, key=lambda row: row.overall_score).scenario_name if rows else ""
        return BenchmarkResponse(
            benchmark_name=request.benchmark_name,
            rows=rows,
            best_scenario_name=best_scenario_name,
        )
