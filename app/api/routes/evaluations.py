from fastapi import APIRouter, HTTPException

from app.models.schemas import EvaluationRequest, EvaluationResponse, EvaluationRunDetails, EvaluationRunSummary
from app.services.evaluation_service import EvaluationService
from app.services.experiment_service import ExperimentService


router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    service = EvaluationService()
    return service.run(request)


@router.get("/runs", response_model=list[EvaluationRunSummary])
def list_evaluation_runs() -> list[EvaluationRunSummary]:
    service = ExperimentService()
    return service.list_run_summaries()


@router.get("/runs/{run_id}", response_model=EvaluationRunDetails)
def get_evaluation_run(run_id: int) -> EvaluationRunDetails:
    service = ExperimentService()
    details = service.get_run_details(run_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Run with id {run_id} was not found.")
    return details
