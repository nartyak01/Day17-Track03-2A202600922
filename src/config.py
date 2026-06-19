from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig

def _provider_from_env(prefix: str = "") -> ProviderConfig:
  key = prefix.upper()
  provider = normalize_provider(os.getenv(f"{key}LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai")))
  model_name = os.getenv(f"{key}LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
  temperature = float(os.getenv(f"{key}LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.2")))
  api_key = None
  base_url = None
  if provider in ("openai", "custom", "openrouter"):
    api_key = os.getenv(f"{key}OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
  if provider == "gemini":
    api_key = os.getenv(f"{key}GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
  if provider == "anthropic":
    api_key = os.getenv(f"{key}ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
  if provider == "custom":
    base_url = os.getenv(f"{key}CUSTOM_BASE_URL", os.getenv("CUSTOM_BASE_URL"))
    api_key = os.getenv(f"{key}CUSTOM_API_KEY", api_key)
  if provider == "ollama":
    base_url = os.getenv(f"{key}OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
  if provider == "openrouter":
    api_key = os.getenv(f"{key}OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
  return ProviderConfig(
    provider=provider,
    model_name=model_name,
    temperature=temperature,
    api_key=api_key,
    base_url=base_url,
  )

def load_config(base_dir: Path | None = None) -> LabConfig:

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(root / ".env")
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "400")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=_provider_from_env(),
        judge_model=_provider_from_env("JUDGE_"),
    )
