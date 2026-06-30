from fastapi import APIRouter

from app.models.schemas import EvaluationRequest, EvaluationResponse
from app.services.evaluation_service import EvaluationService


router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    service = EvaluationService()
    return service.run(request)
