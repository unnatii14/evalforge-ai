import json
from datetime import datetime

from app.db.session import get_connection
from app.models.schemas import EvaluationRunCreate, EvaluationRunRecord


class ExperimentRepository:
    def create_run(self, run: EvaluationRunCreate) -> EvaluationRunRecord:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluation_runs (
                    dataset_name, model_name, chunk_size, chunk_overlap, top_k, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run.dataset_name,
                    run.model_name,
                    run.chunk_size,
                    run.chunk_overlap,
                    run.top_k,
                    json.dumps(run.metadata),
                ),
            )
            connection.commit()
            return EvaluationRunRecord(
                id=cursor.lastrowid,
                created_at=datetime.utcnow(),
                **run.model_dump(),
            )

    def list_runs(self) -> list[EvaluationRunRecord]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, dataset_name, model_name, chunk_size, chunk_overlap, top_k, metadata, created_at
                FROM evaluation_runs
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        records: list[EvaluationRunRecord] = []
        for row in rows:
            payload = json.loads(row["metadata"] or "{}")
            records.append(
                EvaluationRunRecord(
                    id=row["id"],
                    dataset_name=row["dataset_name"],
                    model_name=row["model_name"],
                    chunk_size=row["chunk_size"],
                    chunk_overlap=row["chunk_overlap"],
                    top_k=row["top_k"],
                    metadata=payload,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return records

    def get_run(self, run_id: int) -> EvaluationRunRecord | None:
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT id, dataset_name, model_name, chunk_size, chunk_overlap, top_k, metadata, created_at
                FROM evaluation_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return EvaluationRunRecord(
            id=row["id"],
            dataset_name=row["dataset_name"],
            model_name=row["model_name"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            top_k=row["top_k"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
