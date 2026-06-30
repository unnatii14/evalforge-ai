from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import EvaluationMetricResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> list[EvaluationMetricResult]:
        raise NotImplementedError
