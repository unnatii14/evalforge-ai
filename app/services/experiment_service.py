from app.db.repositories import ExperimentRepository
from app.models.schemas import EvaluationRunCreate, EvaluationRunRecord


class ExperimentService:
    def __init__(self, repository: ExperimentRepository | None = None) -> None:
        self.repository = repository or ExperimentRepository()

    def create_run(self, run: EvaluationRunCreate) -> EvaluationRunRecord:
        return self.repository.create_run(run)

