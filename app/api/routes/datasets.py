from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import DatasetPreview, DatasetUploadResult
from app.services.dataset_service import DatasetService


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/preview", response_model=DatasetPreview)
async def preview_dataset(file: UploadFile = File(...)) -> DatasetPreview:
    service = DatasetService()
    temp_path = Path("data") / "uploads" / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(await file.read())
    return service.preview_csv(temp_path)


@router.post("/persist", response_model=DatasetUploadResult)
async def persist_dataset(file: UploadFile = File(...), dataset_name: str = "uploaded_dataset") -> DatasetUploadResult:
    service = DatasetService()
    temp_path = Path("data") / "uploads" / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(await file.read())
    return service.persist_dataset(temp_path, Path("data") / "datasets", dataset_name)
