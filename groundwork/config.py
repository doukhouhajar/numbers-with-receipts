from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["ollama", "openai-compatible", "openrouter"] = "ollama"
    groundwork_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0 # we want determinism

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
            api_key=s.openai_api_key or "local", # for keyless local servers
            temperature=s.temperature,
        )
    if s.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if not s.openrouter_api_key:
            raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
        return ChatOpenAI(
            model=s.groundwork_model,  # meta-llama/llama-3.3-70b-instruct
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key,
            temperature=s.temperature,
            timeout=60,
            max_retries=2,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r}")


# agent only needs "prompt in, text out"
LLM = Callable[[str], str]
def default_llm(settings: Settings | None = None) -> LLM:
    model = get_model(settings)

    def complete(prompt: str) -> str:
        content = model.invoke(prompt).content
        return content if isinstance(content, str) else str(content)

    return complete

def get_langfuse_handler(settings: Settings | None = None):
    s = settings or get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

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