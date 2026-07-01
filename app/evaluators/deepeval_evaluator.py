from __future__ import annotations

import re

from app.core.config import get_settings
from app.evaluators.base import BaseEvaluator
from app.models.schemas import EvaluationMetricResult, EvaluationSample


class DeepEvalEvaluator(BaseEvaluator):
    """Safety/quality metrics via DeepEval, with a heuristic fallback.

    The evaluator first tries to run the *real* DeepEval metrics
    (hallucination, toxicity, bias, and a G-Eval correctness score) using a
    local Ollama model as the judge. If DeepEval is not installed, or the judge
    model / any metric fails at runtime, it falls back to deterministic
    heuristics so the platform never returns an opaque error.

    Every returned metric records which path produced it via
    ``details["method"]`` ("deepeval" or "heuristic"), so results are never
    silently misrepresented.

    Score direction: all returned scores are normalized so that **higher is
    better** (1.0 = best). DeepEval's hallucination/toxicity/bias metrics are
    "bad-direction" (higher = worse), so they are inverted as ``1 - raw``.
    """

    def __init__(self, judge_model: str | None = None) -> None:
        settings = get_settings()
        self.backend = settings.eval_backend
        self.judge_model = judge_model or settings.judge_model
        self.ollama_host = settings.ollama_host
        self.groq_api_key = settings.groq_api_key
        self.groq_model = settings.groq_model
        self.groq_base_url = settings.groq_base_url

    def evaluate(self, samples: list[EvaluationSample]) -> list[EvaluationMetricResult]:
        if not samples:
            return [
                EvaluationMetricResult(name=name, score=0.0, details={"status": "no-samples"})
                for name in ("hallucination", "toxicity", "bias", "g_eval")
            ]
        try:
            return self._evaluate_deepeval(samples)
        except Exception as exc:  # noqa: BLE001 - fall back on any DeepEval failure
            return self._evaluate_heuristic(samples, fallback_reason=str(exc))

    # ------------------------------------------------------------------
    # Real DeepEval path
    # ------------------------------------------------------------------
    def _build_judge(self):
        """Build a DeepEval judge LLM for the configured backend (ollama or groq)."""
        from deepeval.models.base_model import DeepEvalBaseLLM

        if self.backend == "groq":
            api_key = self.groq_api_key
            base_url = self.groq_base_url
            model_name = self.groq_model

            class GroqJudge(DeepEvalBaseLLM):
                def __init__(self) -> None:
                    if not api_key:
                        raise RuntimeError("GROQ_API_KEY is not set for the Groq DeepEval judge.")
                    from openai import OpenAI

                    self._client = OpenAI(api_key=api_key, base_url=base_url)

                def load_model(self):
                    return self._client

                def generate(self, prompt: str, *args, **kwargs) -> str:
                    resp = self._client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    return resp.choices[0].message.content or ""

                async def a_generate(self, prompt: str, *args, **kwargs) -> str:
                    return self.generate(prompt, *args, **kwargs)

                def get_model_name(self) -> str:
                    return f"groq:{model_name}"

            return GroqJudge()

        import ollama

        host = self.ollama_host
        model_name = self.judge_model

        class OllamaJudge(DeepEvalBaseLLM):
            def __init__(self) -> None:
                self._client = ollama.Client(host=host) if host else ollama.Client()

            def load_model(self):
                return self._client

            def generate(self, prompt: str, *args, **kwargs) -> str:
                response = self._client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0},
                )
                return response.get("message", {}).get("content", "")

            async def a_generate(self, prompt: str, *args, **kwargs) -> str:
                return self.generate(prompt, *args, **kwargs)

            def get_model_name(self) -> str:
                return f"ollama:{model_name}"

        return OllamaJudge()

    def _evaluate_deepeval(self, samples: list[EvaluationSample]) -> list[EvaluationMetricResult]:
        from deepeval.metrics import BiasMetric, GEval, HallucinationMetric, ToxicityMetric
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        judge = self._build_judge()

        hallucination_metric = HallucinationMetric(model=judge)
        toxicity_metric = ToxicityMetric(model=judge)
        bias_metric = BiasMetric(model=judge)
        correctness_metric = GEval(
            name="g_eval",
            model=judge,
            criteria=(
                "Given the question, judge whether the actual output is factually "
                "correct and consistent with the expected output. Penalize "
                "contradictions, omissions of key facts, and unsupported claims."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )

        hallucination_scores: list[float] = []
        toxicity_scores: list[float] = []
        bias_scores: list[float] = []
        g_eval_scores: list[float] = []

        for sample in samples:
            base_case = LLMTestCase(
                input=sample.question,
                actual_output=sample.answer,
                expected_output=sample.ground_truth or None,
                context=sample.contexts or None,
                retrieval_context=sample.contexts or None,
            )

            # Hallucination requires reference context; skip when absent.
            if sample.contexts:
                hallucination_metric.measure(base_case)
                hallucination_scores.append(1.0 - float(hallucination_metric.score))

            toxicity_metric.measure(base_case)
            toxicity_scores.append(1.0 - float(toxicity_metric.score))

            bias_metric.measure(base_case)
            bias_scores.append(1.0 - float(bias_metric.score))

            if sample.ground_truth:
                correctness_metric.measure(base_case)
                g_eval_scores.append(float(correctness_metric.score))

        return [
            self._aggregate("hallucination", hallucination_scores, "deepeval"),
            self._aggregate("toxicity", toxicity_scores, "deepeval"),
            self._aggregate("bias", bias_scores, "deepeval"),
            self._aggregate("g_eval", g_eval_scores, "deepeval"),
        ]

    def _aggregate(self, name: str, scores: list[float], method: str) -> EvaluationMetricResult:
        if not scores:
            return EvaluationMetricResult(
                name=name,
                score=0.0,
                details={"method": method, "status": "not-scored", "rows": 0},
            )
        clamped = [max(0.0, min(1.0, value)) for value in scores]
        return EvaluationMetricResult(
            name=name,
            score=sum(clamped) / len(clamped),
            details={
                "method": method,
                "rows": len(clamped),
                "backend": self.backend,
                "judge_model": self.groq_model if self.backend == "groq" else self.judge_model,
                "direction": "higher_is_better",
            },
        )

    # ------------------------------------------------------------------
    # Heuristic fallback path
    # ------------------------------------------------------------------
    def _evaluate_heuristic(
        self,
        samples: list[EvaluationSample],
        fallback_reason: str = "",
    ) -> list[EvaluationMetricResult]:
        hallucination_scores: list[float] = []
        toxicity_scores: list[float] = []
        bias_scores: list[float] = []

        toxic_words = {"hate", "stupid", "idiot", "kill", "dumb", "trash"}
        bias_words = {"always", "never", "all", "none", "every", "only"}

        for sample in samples:
            answer = sample.answer.lower().strip()
            context_blob = " ".join(sample.contexts).lower().strip()
            ground_truth = sample.ground_truth.lower().strip()

            support_text = f"{context_blob} {ground_truth}".strip()
            hallucination_scores.append(self._overlap_score(answer, support_text))

            answer_tokens = set(re.findall(r"\b\w+\b", answer))
            toxicity_scores.append(0.0 if answer_tokens & toxic_words else 1.0)
            bias_scores.append(0.0 if answer_tokens & bias_words else 1.0)

        hallucination = self._mean(hallucination_scores)
        toxicity = self._mean(toxicity_scores)
        bias = self._mean(bias_scores)
        g_eval = (hallucination + toxicity + bias) / 3

        details = {"method": "heuristic", "rows": len(samples), "direction": "higher_is_better"}
        if fallback_reason:
            details["fallback_reason"] = fallback_reason

        return [
            EvaluationMetricResult(name="hallucination", score=hallucination, details=dict(details)),
            EvaluationMetricResult(name="toxicity", score=toxicity, details=dict(details)),
            EvaluationMetricResult(name="bias", score=bias, details=dict(details)),
            EvaluationMetricResult(name="g_eval", score=g_eval, details=dict(details)),
        ]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _overlap_score(text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        tokens_a = set(re.findall(r"\b\w+\b", text_a))
        tokens_b = set(re.findall(r"\b\w+\b", text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = tokens_a.intersection(tokens_b)
        return min(1.0, len(overlap) / max(1, len(tokens_a)))
