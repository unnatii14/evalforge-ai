from __future__ import annotations

from app.core.config import get_settings
from app.llm_providers.base import BaseLLMProvider
from app.llm_providers.ollama_provider import OllamaProvider
from app.llm_providers.groq_provider import GroqProvider
from app.models.schemas import LLMConfig, ProviderName


class ProviderNotImplementedError(NotImplementedError):
    """Raised when a configured provider has no implementation yet."""


def get_provider(config: LLMConfig) -> BaseLLMProvider:
    """Return the provider implementation for the requested provider.

    The ``ProviderName`` enum advertises several providers; only those with a
    real implementation are returned. Requesting an unimplemented provider
    raises a clear error instead of silently falling back to Ollama, so the
    enum never misrepresents what the platform can actually do.
    """
    settings = get_settings()

    if config.provider == ProviderName.ollama:
        return OllamaProvider(host=settings.ollama_host)

    if config.provider == ProviderName.groq:
        return GroqProvider(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
        )

    raise ProviderNotImplementedError(
        f"Provider '{config.provider.value}' is declared but not implemented yet. "
        "Currently supported providers: ollama, groq."
    )
