from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import LLMConfig, LLMGenerationResult


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: LLMConfig, system_prompt: str | None = None) -> LLMGenerationResult:
        raise NotImplementedError
