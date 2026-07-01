import json
from datetime import datetime
from statistics import mean

from app.db.session import get_connection
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRunCreate,
    EvaluationRunRecord,
    EvaluationRunSummary,
    EvaluationSampleRecord,
)


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

    def list_run_summaries(self) -> list[EvaluationRunSummary]:
        runs = self.list_runs()
        summaries: list[EvaluationRunSummary] = []
        for run in runs:
            metrics = self.list_metrics(run.id)
            overall_score = mean([metric.score for metric in metrics]) if metrics else 0.0
            summaries.append(
                EvaluationRunSummary(
                    run=run,
                    metrics=metrics,
                    overall_score=overall_score,
                )
            )
        return summaries

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

    def save_metrics(self, run_id: int, metrics: list[EvaluationMetricResult]) -> None:
        if not metrics:
            return
        payload = [
            (
                run_id,
                metric.name,
                metric.score,
                json.dumps(metric.details),
            )
            for metric in metrics
        ]
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO evaluation_metrics (run_id, metric_name, score, details)
                VALUES (?, ?, ?, ?)
                """,
                payload,
            )
            connection.commit()

    def list_metrics(self, run_id: int) -> list[EvaluationMetricResult]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT metric_name, score, details
                FROM evaluation_metrics
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            EvaluationMetricResult(
                name=row["metric_name"],
                score=float(row["score"]),
                details=json.loads(row["details"] or "{}"),
            )
            for row in rows
        ]

    def save_samples(self, run_id: int, samples: list[EvaluationSampleRecord]) -> None:
        if not samples:
            return
        payload = [
            (
                run_id,
                sample.question,
                sample.answer,
                json.dumps(sample.contexts),
                sample.ground_truth,
                sample.answer_source,
            )
            for sample in samples
        ]
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO evaluation_samples (run_id, question, answer, contexts, ground_truth, answer_source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            connection.commit()

    def list_samples(self, run_id: int) -> list[EvaluationSampleRecord]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT question, answer, contexts, ground_truth, answer_source
                FROM evaluation_samples
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            EvaluationSampleRecord(
                question=row["question"],
                answer=row["answer"],
                contexts=json.loads(row["contexts"] or "[]"),
                ground_truth=row["ground_truth"],
                answer_source=row["answer_source"],
            )
            for row in rows
        ]
