"""Persistence layer + FK integrity tests."""
from __future__ import annotations

from app.db.repositories import ExperimentRepository
from app.db.session import get_connection
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRunCreate,
    EvaluationSampleRecord,
)


def _make_run(**overrides) -> EvaluationRunCreate:
    payload = dict(dataset_name="demo", model_name="llama3.2", metadata={"provider": "ollama"})
    payload.update(overrides)
    return EvaluationRunCreate(**payload)


def test_create_and_get_run_roundtrip():
    repo = ExperimentRepository()
    created = repo.create_run(_make_run())
    fetched = repo.get_run(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.dataset_name == "demo"
    assert fetched.metadata["provider"] == "ollama"


def test_metrics_and_samples_persist_and_summarize():
    repo = ExperimentRepository()
    run = repo.create_run(_make_run())
    repo.save_metrics(
        run.id,
        [
            EvaluationMetricResult(name="faithfulness", score=0.9, details={}),
            EvaluationMetricResult(name="answer_relevancy", score=0.7, details={}),
        ],
    )
    repo.save_samples(
        run.id,
        [
            EvaluationSampleRecord(
                question="q", answer="a", contexts=["c"], ground_truth="g", answer_source="provided"
            )
        ],
    )

    metrics = repo.list_metrics(run.id)
    assert {m.name for m in metrics} == {"faithfulness", "answer_relevancy"}

    summaries = repo.list_run_summaries()
    assert len(summaries) == 1
    # overall = mean(0.9, 0.7) = 0.8
    assert abs(summaries[0].overall_score - 0.8) < 1e-9

    samples = repo.list_samples(run.id)
    assert len(samples) == 1 and samples[0].answer_source == "provided"


def test_get_missing_run_returns_none():
    assert ExperimentRepository().get_run(9999) is None


def test_foreign_key_cascade_delete():
    repo = ExperimentRepository()
    run = repo.create_run(_make_run())
    repo.save_metrics(run.id, [EvaluationMetricResult(name="faithfulness", score=1.0, details={})])
    repo.save_samples(
        run.id,
        [EvaluationSampleRecord(question="q", answer="a", contexts=[], ground_truth="g", answer_source="generated")],
    )

    conn = get_connection()
    conn.execute("DELETE FROM evaluation_runs WHERE id = ?", (run.id,))
    conn.commit()

    assert repo.list_metrics(run.id) == []
    assert repo.list_samples(run.id) == []


def test_pragma_foreign_keys_enabled():
    value = get_connection().execute("PRAGMA foreign_keys").fetchone()[0]
    assert value == 1
