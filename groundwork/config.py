from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["ollama", "openai-compatible"] = "ollama"
    groundwork_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str | None = None          # http://localhost:8000/v1 for vLLM
    openai_api_key: str | None = None
    temperature: float = 0.0                    # grounded reasoning wants determinism

    # Sandbox 
    e2b_api_key: str | None = None

    # Observability 
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    groundwork_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_model(settings: Settings | None = None):
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.groundwork_model,
            base_url=s.ollama_base_url,
            temperature=s.temperature,
        )
    if s.llm_provider == "openai-compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.groundwork_model,
            base_url=s.openai_base_url,
            api_key=s.openai_api_key or "local",   # placeholder for keyless local servers
            temperature=s.temperature,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r}")


def get_langfuse_handler(settings: Settings | None = None):
    s = settings or get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # v3: initialise the client once. pydantic-settings reads .env into Settings
    # but does NOT export to os.environ, so we pass the keys explicitly.
    Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )
    return CallbackHandler()


def setup_logging(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, s.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )