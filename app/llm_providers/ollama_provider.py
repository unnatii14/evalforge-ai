from __future__ import annotations

import time

import ollama

from app.llm_providers.base import BaseLLMProvider
from app.models.schemas import LLMConfig, LLMGenerationResult, ProviderName


class OllamaProvider(BaseLLMProvider):
    def __init__(self, host: str | None = None) -> None:
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def generate(self, prompt: str, config: LLMConfig, system_prompt: str | None = None) -> LLMGenerationResult:
        start_time = time.perf_counter()
        response = self.client.chat(
            model=config.model_name,
            messages=[
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            },
        )
        latency_seconds = time.perf_counter() - start_time
        message = response.get("message", {})
        return LLMGenerationResult(
            provider=ProviderName.ollama,
            model_name=config.model_name,
            output=message.get("content", ""),
            latency_seconds=latency_seconds,
        )
