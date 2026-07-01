"""EvaluationService orchestration, provider routing, and integrity tests."""
from __future__ import annotations

import pytest

from app.llm_providers.factory import ProviderNotImplementedError, get_provider
from app.llm_providers.groq_provider import GroqProvider
from app.llm_providers.ollama_provider import OllamaProvider
from app.models.schemas import (
    EvaluationRequest,
    EvaluationSample,
    LLMConfig,
    ProviderName,
)
from app.services.evaluation_service import EvaluationService
from app.services.experiment_service import ExperimentService
from tests.conftest import FakeProvider, FakeRagasEvaluator


def _service(provider=None):
    from app.evaluators.deepeval_evaluator import DeepEvalEvaluator

    return EvaluationService(
        provider=provider or FakeProvider(),
        ragas_evaluator=FakeRagasEvaluator(),
        deepeval_evaluator=DeepEvalEvaluator(),
    )


def _request(**overrides):
    payload = dict(
        dataset_name="demo",
        config=LLMConfig(model_name="llama3.2"),
        metrics=["ragas", "deepeval"],
        samples=[
            EvaluationSample(
                question="Capital of France?",
                answer="Paris is the capital.",
                contexts=["Paris is the capital of France."],
                ground_truth="Paris",
            ),
            EvaluationSample(question="2+2?", answer="", contexts=["4"], ground_truth="4"),
        ],
    )
    payload.update(overrides)
    return EvaluationRequest(**payload)


def test_provider_factory_routes_ollama():
    assert isinstance(get_provider(LLMConfig(provider=ProviderName.ollama)), OllamaProvider)


def test_provider_factory_routes_groq():
    # Constructed without an API key (lazy client); routing must still work.
    assert isinstance(get_provider(LLMConfig(provider=ProviderName.groq)), GroqProvider)


@pytest.mark.parametrize("provider", [ProviderName.openai, ProviderName.gemini])
def test_provider_factory_rejects_unimplemented(provider):
    with pytest.raises(ProviderNotImplementedError):
        get_provider(LLMConfig(provider=provider))


def test_end_to_end_run_persists_and_returns_metrics():
    resp = _service().run(_request())
    assert resp.run.id is not None
    names = {m.name for m in resp.metrics}
    assert {"faithfulness", "hallucination", "toxicity", "bias", "g_eval"} <= names

    details = ExperimentService().get_run_details(resp.run.id)
    assert details is not None and len(details.samples) == 2


def test_answer_source_tracked_provided_and_generated():
    resp = _service().run(_request())
    details = ExperimentService().get_run_details(resp.run.id)
    sources = {s.answer_source for s in details.samples}
    assert sources == {"provided", "generated"}


def test_generation_failure_does_not_inflate_answer():
    svc = _service(provider=FakeProvider(fail=True))
    req = _request(
        samples=[EvaluationSample(question="q", answer="", contexts=["c"], ground_truth="SECRET_TRUTH")]
    )
    generated, sources, latency = svc._generate_answers(req, req.samples)
    assert generated[0].answer == ""  # not copied from ground truth
    assert generated[0].ground_truth == "SECRET_TRUTH"
    assert sources == ["generation_failed"]


def test_deepeval_results_are_labeled():
    resp = _service().run(_request(metrics=["deepeval"]))
    safety = [m for m in resp.metrics if m.name in {"hallucination", "toxicity", "bias", "g_eval"}]
    assert safety and all(m.details.get("method") in {"deepeval", "heuristic"} for m in safety)
