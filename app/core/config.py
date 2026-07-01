from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from the environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "EvalForge"
    environment: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data") / "evalforge.db"

    # Local model configuration used by the RAGAS and DeepEval evaluators.
    # These default to Ollama so the platform runs fully locally without an
    # OpenAI API key. Override via environment variables if needed.
    ollama_host: str | None = None
    judge_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    # Evaluation backend selector: "ollama" (local) or "groq" (free hosted API).
    # Groq is used for cloud deployments (e.g. Hugging Face Spaces) where a local
    # Ollama server is not available. Groq has no embeddings endpoint, so RAGAS
    # embeddings fall back to a small local Hugging Face model in that mode.
    eval_backend: str = "ollama"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
