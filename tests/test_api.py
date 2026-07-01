"""API route tests via FastAPI TestClient (offline)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.db.repositories import ExperimentRepository
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRunCreate,
    EvaluationSampleRecord,
)

client = TestClient(app)


def _seed():
    repo = ExperimentRepository()
    run = repo.create_run(
        EvaluationRunCreate(dataset_name="demo", model_name="llama3.2", metadata={"provider": "ollama"})
    )
    repo.save_metrics(run.id, [EvaluationMetricResult(name="faithfulness", score=0.9, details={})])
    repo.save_samples(
        run.id,
        [EvaluationSampleRecord(question="q", answer="a", contexts=["c"], ground_truth="g", answer_source="provided")],
    )
    return run


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_runs_endpoint():
    run = _seed()
    resp = client.get("/evaluations/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert any(item["run"]["id"] == run.id for item in body)


def test_get_run_detail_endpoint():
    run = _seed()
    resp = client.get(f"/evaluations/runs/{run.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run"]["id"] == run.id
    assert len(body["samples"]) == 1


def test_get_missing_run_returns_404():
    resp = client.get("/evaluations/runs/999999")
    assert resp.status_code == 404


def test_run_evaluation_endpoint_offline():
    # Provided answers + deepeval-only metric => no provider/network needed;
    # deepeval is absent so the evaluator uses its labeled heuristic fallback.
    payload = {
        "dataset_name": "demo",
        "config": {"provider": "ollama", "model_name": "llama3.2"},
        "metrics": ["deepeval"],
        "samples": [
            {"question": "Capital of France?", "answer": "Paris.", "contexts": ["Paris is the capital."], "ground_truth": "Paris"}
        ],
    }
    resp = client.post("/evaluations/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["run"]["id"] is not None
    metric_names = {m["name"] for m in body["metrics"]}
    assert {"hallucination", "toxicity", "bias", "g_eval"} <= metric_names
