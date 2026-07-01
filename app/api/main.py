from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.benchmarks import router as benchmarks_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.reports import router as reports_router
from app.core.config import get_settings
from app.db.session import initialize_database


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(datasets_router)
app.include_router(evaluations_router)
app.include_router(benchmarks_router)
app.include_router(reports_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}
