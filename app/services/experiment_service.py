from app.db.repositories import ExperimentRepository
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRunCreate,
    EvaluationRunDetails,
    EvaluationRunRecord,
    EvaluationRunSummary,
    EvaluationSampleRecord,
)


class ExperimentService:
    def __init__(self, repository: ExperimentRepository | None = None) -> None:
        self.repository = repository or ExperimentRepository()

    def create_run(self, run: EvaluationRunCreate) -> EvaluationRunRecord:
        return self.repository.create_run(run)

    def save_metrics(self, run_id: int, metrics: list[EvaluationMetricResult]) -> None:
        self.repository.save_metrics(run_id, metrics)

    def save_samples(self, run_id: int, samples: list[EvaluationSampleRecord]) -> None:
        self.repository.save_samples(run_id, samples)

    def get_run_details(self, run_id: int) -> EvaluationRunDetails | None:
        run = self.repository.get_run(run_id)
        if run is None:
            return None
        return EvaluationRunDetails(
            run=run,
            metrics=self.repository.list_metrics(run_id),
            samples=self.repository.list_samples(run_id),
        )

    def list_run_summaries(self) -> list[EvaluationRunSummary]:
        return self.repository.list_run_summaries()

