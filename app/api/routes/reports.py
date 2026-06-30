from fastapi import APIRouter

from app.models.schemas import ReportRequest, ReportResult
from app.reports.report_service import ReportService


router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResult)
def generate_report(request: ReportRequest) -> ReportResult:
    service = ReportService()
    return service.generate(request)
