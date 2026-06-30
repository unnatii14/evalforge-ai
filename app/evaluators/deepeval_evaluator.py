from __future__ import annotations

from app.evaluators.base import BaseEvaluator
from app.models.schemas import EvaluationMetricResult


class DeepEvalEvaluator(BaseEvaluator):
    def evaluate(self, *args, **kwargs) -> list[EvaluationMetricResult]:
        try:
            from deepeval.metrics import BiasMetric, GEval, HallucinationMetric, ToxicityMetric
        except ImportError as exc:
            raise RuntimeError("DeepEval is not available in the current environment.") from exc

        _ = (BiasMetric, GEval, HallucinationMetric, ToxicityMetric)
        return [
            EvaluationMetricResult(name="hallucination", score=0.0, details={"status": "placeholder"}),
            EvaluationMetricResult(name="toxicity", score=0.0, details={"status": "placeholder"}),
            EvaluationMetricResult(name="bias", score=0.0, details={"status": "placeholder"}),
            EvaluationMetricResult(name="g_eval", score=0.0, details={"status": "placeholder"}),
        ]
