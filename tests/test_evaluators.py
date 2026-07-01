"""Evaluator + dataset service unit/integration tests (offline)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.evaluators.deepeval_evaluator import DeepEvalEvaluator
from app.models.schemas import EvaluationSample
from app.services.dataset_service import DatasetService, DatasetValidationError


def _samples():
    return [
        EvaluationSample(
            question="Capital of France?",
            answer="Paris is the capital.",
            contexts=["Paris is the capital of France."],
            ground_truth="Paris",
        ),
        EvaluationSample(
            question="2+2?",
            answer="you are an idiot, always wrong",
            contexts=["4"],
            ground_truth="4",
        ),
    ]


def test_deepeval_falls_back_to_heuristic_when_deepeval_absent():
    # deepeval is not installed in the test env, so the real path raises and
    # the evaluator must fall back to labeled heuristics.
    results = DeepEvalEvaluator().evaluate(_samples())
    assert {r.name for r in results} == {"hallucination", "toxicity", "bias", "g_eval"}
    assert all(r.details.get("method") == "heuristic" for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_deepeval_scores_penalize_toxic_and_biased_answer():
    results = {r.name: r.score for r in DeepEvalEvaluator().evaluate(_samples())}
    # one clean answer, one toxic+biased -> averaged 0.5 on both axes
    assert results["toxicity"] == 0.5
    assert results["bias"] == 0.5


def test_deepeval_empty_samples():
    results = DeepEvalEvaluator().evaluate([])
    assert all(r.details.get("status") == "no-samples" for r in results)


def test_dataset_validation_requires_question_column():
    df = pd.DataFrame({"not_question": ["x"]})
    with pytest.raises(DatasetValidationError):
        DatasetService().validate_dataframe(df)


def test_dataset_load_rows_and_context_parsing(tmp_path, monkeypatch):
    csv = tmp_path / "demo.csv"
    pd.DataFrame(
        {
            "question": ["What is X?"],
            "ground_truth": ["X is a thing."],
            "context": ["Context about X."],
            "expected_answer": ["X is a thing."],
        }
    ).to_csv(csv, index=False)

    service = DatasetService()
    monkeypatch.setattr(service, "resolve_dataset_path", lambda name: csv)
    rows = service.load_dataset_rows("demo", limit=5)
    assert len(rows) == 1
    assert rows[0].question == "What is X?"
    assert rows[0].contexts == ["Context about X."]
    assert rows[0].ground_truth == "X is a thing."
