from fastapi import APIRouter

from app.models.schemas import BenchmarkRequest, BenchmarkResponse
from app.services.benchmark_service import BenchmarkService


router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.post("/run", response_model=BenchmarkResponse)
def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    service = BenchmarkService()
    return service.run(request)
