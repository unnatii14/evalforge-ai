from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.models.schemas import DatasetPreview, DatasetRow, DatasetUploadResult


class DatasetValidationError(ValueError):
    pass


class DatasetService:
    required_columns = ["question"]
    optional_columns = ["ground_truth", "context", "expected_answer"]

    def validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        missing_columns = [column for column in self.required_columns if column not in dataframe.columns]
        if missing_columns:
            raise DatasetValidationError(f"Missing required dataset columns: {', '.join(missing_columns)}")

    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        dataframe = pd.read_csv(file_path)
        self.validate_dataframe(dataframe)
        return dataframe

    def preview_csv(self, file_path: str | Path, sample_size: int = 5) -> DatasetPreview:
        dataframe = self.load_csv(file_path)
        sample_rows = [
            DatasetRow(**row).model_dump()
            for row in dataframe.head(sample_size).to_dict(orient="records")
        ]
        return DatasetPreview(
            row_count=len(dataframe),
            columns=list(dataframe.columns),
            sample_rows=[DatasetRow(**row) for row in dataframe.head(sample_size).to_dict(orient="records")],
        )

    def persist_dataset(self, source_path: str | Path, destination_dir: str | Path, dataset_name: str) -> DatasetUploadResult:
        dataframe = self.load_csv(source_path)
        destination_directory = Path(destination_dir)
        destination_directory.mkdir(parents=True, exist_ok=True)

        target_path = destination_directory / f"{dataset_name}.csv"
        dataframe.to_csv(target_path, index=False)

        return DatasetUploadResult(
            dataset_name=dataset_name,
            row_count=len(dataframe),
            columns=list(dataframe.columns),
            file_path=str(target_path),
        )
