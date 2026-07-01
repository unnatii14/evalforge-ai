from __future__ import annotations

import time

from app.core.config import get_settings
from app.llm_providers.base import BaseLLMProvider
from app.models.schemas import LLMConfig, LLMGenerationResult, ProviderName


class GroqProvider(BaseLLMProvider):
    """LLM provider backed by Groq's free, OpenAI-compatible API.

    Used for cloud deployments (e.g. Hugging Face Spaces) where a local Ollama
    server is unavailable. The client is created lazily so the provider can be
    constructed without an API key present (e.g. in tests); the key is only
    required when ``generate`` is actually called.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.base_url = base_url or settings.groq_base_url
        self.default_model = model or settings.groq_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it as an environment variable / "
                    "Hugging Face Space secret to use the Groq backend."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, prompt: str, config: LLMConfig, system_prompt: str | None = None) -> LLMGenerationResult:
        # A model name of "ollama"-style defaults doesn't exist on Groq, so prefer
        # the configured Groq model unless the caller passed an explicit Groq model.
        model = config.model_name if config.provider == ProviderName.groq and config.model_name else self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        latency = time.perf_counter() - start
        output = response.choices[0].message.content or ""
        return LLMGenerationResult(
            provider=ProviderName.groq,
            model_name=model,
            output=output,
            latency_seconds=latency,
        )
