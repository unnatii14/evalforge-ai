from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from app.models.schemas import DatasetPreview, DatasetRow, DatasetUploadResult, EvaluationSample


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

    def resolve_dataset_path(self, dataset_name: str) -> Path:
        candidate_paths = [
            Path("data") / "datasets" / f"{dataset_name}.csv",
            Path("evaluation") / f"{dataset_name}.csv",
        ]
        for path in candidate_paths:
            if path.exists():
                return path
        raise FileNotFoundError(f"Dataset '{dataset_name}' was not found in data/datasets or evaluation.")

    def load_dataset_rows(self, dataset_name: str, limit: int = 10) -> list[EvaluationSample]:
        dataset_path = self.resolve_dataset_path(dataset_name)
        dataframe = self.load_csv(dataset_path).head(limit)
        records: list[EvaluationSample] = []
        for row in dataframe.to_dict(orient="records"):
            normalized: dict[str, Any] = {
                key: (None if pd.isna(value) else value) for key, value in row.items()
            }
            contexts = self._extract_contexts(normalized)
            answer_value = normalized.get("expected_answer") or ""
            ground_truth_value = normalized.get("ground_truth") or ""
            records.append(
                EvaluationSample(
                    question=str(normalized.get("question", "")),
                    answer=str(answer_value),
                    contexts=contexts,
                    ground_truth=str(ground_truth_value),
                )
            )
        return records

    @staticmethod
    def _extract_contexts(row: dict[str, Any]) -> list[str]:
        if row.get("contexts"):
            raw_contexts = row.get("contexts")
            if isinstance(raw_contexts, list):
                return [str(item) for item in raw_contexts if str(item).strip()]
            if isinstance(raw_contexts, str):
                try:
                    parsed = ast.literal_eval(raw_contexts)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed if str(item).strip()]
                except (ValueError, SyntaxError):
                    pass
                return [raw_contexts] if raw_contexts.strip() else []

        context_value = row.get("context")
        return [str(context_value)] if context_value else []

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
