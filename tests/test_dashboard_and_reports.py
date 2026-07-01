"""Dashboard data mapping + report generation tests."""
from __future__ import annotations

import os.path as osp

from app.dashboard.data_loader import DashboardDataLoader
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRunCreate,
    EvaluationSampleRecord,
    ReportFormat,
    ReportRequest,
)
from app.db.repositories import ExperimentRepository
from app.reports.report_service import ReportService
from app.services.evaluation_service import EvaluationService
from app.services.experiment_service import ExperimentService
from tests.conftest import FakeProvider, FakeRagasEvaluator


def _seed_run():
    repo = ExperimentRepository()
    run = repo.create_run(
        EvaluationRunCreate(dataset_name="demo", model_name="llama3.2", metadata={"provider": "ollama"})
    )
    repo.save_metrics(
        run.id,
        [
            EvaluationMetricResult(name="faithfulness", score=0.9, details={}),
            EvaluationMetricResult(name="hallucination", score=0.5, details={"method": "heuristic"}),
        ],
    )
    repo.save_samples(
        run.id,
        [EvaluationSampleRecord(question="q", answer="a", contexts=["c"], ground_truth="g", answer_source="provided")],
    )
    return run


def test_dashboard_get_runs_maps_metrics():
    run = _seed_run()
    loader = DashboardDataLoader()
    rows = loader.get_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == run.id
    assert row["faithfulness"] == 0.9
    assert row["hallucination"] == 0.5
    assert row["provider"] == "ollama"


def test_dashboard_get_run_details_dataframe():
    run = _seed_run()
    df = DashboardDataLoader().get_run_details(run.id)
    assert not df.empty
    assert list(df["question"]) == ["q"]
    assert df.iloc[0]["faithfulness"] == 0.9


def _eval_response():
    svc = EvaluationService(provider=FakeProvider(), ragas_evaluator=FakeRagasEvaluator())
    from app.models.schemas import EvaluationRequest, EvaluationSample, LLMConfig

    return svc.run(
        EvaluationRequest(
            dataset_name="demo",
            config=LLMConfig(model_name="llama3.2"),
            metrics=["ragas"],
            samples=[EvaluationSample(question="q", answer="a", contexts=["c"], ground_truth="g")],
        )
    )


def test_report_csv(tmp_path):
    resp = _eval_response()
    result = ReportService(output_dir=tmp_path).generate(
        ReportRequest(report_name="r_csv", format=ReportFormat.csv, evaluation=resp)
    )
    assert osp.exists(result.file_path) and result.row_count > 0


def test_report_json(tmp_path):
    resp = _eval_response()
    result = ReportService(output_dir=tmp_path).generate(
        ReportRequest(report_name="r_json", format=ReportFormat.json, evaluation=resp)
    )
    assert osp.exists(result.file_path)


def test_report_pdf(tmp_path):
    resp = _eval_response()
    result = ReportService(output_dir=tmp_path).generate(
        ReportRequest(report_name="r_pdf", format=ReportFormat.pdf, evaluation=resp)
    )
    assert osp.exists(result.file_path)
