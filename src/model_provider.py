from __future__ import annotations

from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "gemini": "gemini",
        "open-ai": "openai",
    }
    key = value.strip().lower()
    return aliases.get(key, key)


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    if provider == "openai":
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    if provider == "custom":
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    if provider == "ollama":
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )
    if provider == "openrouter":
        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    raise ValueError(f"Unsupported provider: {config.provider}")
