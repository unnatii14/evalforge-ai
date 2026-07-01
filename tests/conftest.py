"""Shared fixtures for the EvalForge integration test suite.

The suite runs fully offline: settings are pointed at a throwaway SQLite
database per test, the LLM provider is mocked, and RAGAS is faked so no
network / Ollama / OpenAI access is required.
"""
from __future__ import annotations

import pytest

from app.models.schemas import (
    EvaluationMetricResult,
    LLMGenerationResult,
    ProviderName,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point application settings at an isolated temp DB for each test."""
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("sqlite_path", str(tmp_path / "evalforge.db"))

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.db.session import initialize_database

    initialize_database()
    yield tmp_path
    get_settings.cache_clear()


class FakeProvider:
    """Deterministic provider stand-in that never touches the network."""

    def __init__(self, output: str = "A generated answer.", fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.calls = 0

    def generate(self, prompt, config, system_prompt=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated provider failure")
        return LLMGenerationResult(
            provider=ProviderName.ollama,
            model_name=config.model_name,
            output=self.output,
            latency_seconds=0.01,
        )


class FakeRagasEvaluator:
    """Returns fixed RAGAS-style metrics without importing ragas."""

    def evaluate(self, inputs):
        return [
            EvaluationMetricResult(name="faithfulness", score=0.9, details={"rows": len(inputs.questions)}),
            EvaluationMetricResult(name="answer_relevancy", score=0.8, details={}),
            EvaluationMetricResult(name="context_recall", score=0.7, details={}),
            EvaluationMetricResult(name="context_precision", score=0.6, details={}),
            EvaluationMetricResult(name="answer_correctness", score=0.85, details={}),
        ]


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def fake_ragas():
    return FakeRagasEvaluator()
