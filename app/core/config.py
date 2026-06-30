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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
