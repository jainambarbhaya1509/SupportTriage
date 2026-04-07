"""OpenAI client helpers used by the environment, inference script, and mail tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model_name: str
    api_key: str


def default_model_name() -> str:
    return (
        os.getenv("MODEL_NAME", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or DEFAULT_OPENAI_MODEL
    )


def resolve_openai_config(*, model_override: str | None = None) -> LLMConfig | None:
    api_key = os.getenv("HF_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return LLMConfig(
        provider="openai",
        base_url=os.getenv("API_BASE_URL", "").strip() or DEFAULT_OPENAI_BASE_URL,
        model_name=model_override or default_model_name(),
        api_key=api_key,
    )


def create_openai_client(config: LLMConfig) -> OpenAI:
    return OpenAI(base_url=config.base_url, api_key=config.api_key)


def chat_completion(
    config: LLMConfig,
    *,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 180,
) -> str:
    client = create_openai_client(config)
    completion = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""
