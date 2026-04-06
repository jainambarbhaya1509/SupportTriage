"""Helpers for OpenAI-compatible LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROK_MODEL = "grok-3-mini-beta"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model_name: str
    api_key: str


def normalize_provider_hint(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"grok", "xai"}:
        return "grok"
    if normalized in {"openai", "groq", "compatible"}:
        return normalized
    raise ValueError(f"Unsupported provider hint: {provider}")


def provider_label_from_base_url(base_url: str) -> str:
    host = urlparse(base_url).netloc.lower()
    if "groq.com" in host:
        return "groq"
    if "x.ai" in host:
        return "grok"
    if "openai.com" in host:
        return "openai"
    return "compatible"


def default_model_name(preferred_provider: str | None = None) -> str:
    hint = normalize_provider_hint(preferred_provider)
    if hint == "openai":
        return (
            os.getenv("MODEL_NAME", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or DEFAULT_OPENAI_MODEL
        )
    if hint == "groq":
        return (
            os.getenv("MODEL_NAME", "").strip()
            or os.getenv("GROQ_MODEL", "").strip()
            or DEFAULT_GROQ_MODEL
        )
    if hint == "grok":
        return (
            os.getenv("MODEL_NAME", "").strip()
            or os.getenv("XAI_MODEL", "").strip()
            or DEFAULT_GROK_MODEL
        )
    return (
        os.getenv("MODEL_NAME", "").strip()
        or os.getenv("GROQ_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or DEFAULT_GROQ_MODEL
    )


def resolve_llm_config(
    preferred_provider: str | None = None,
    *,
    model_override: str | None = None,
) -> LLMConfig | None:
    hint = normalize_provider_hint(preferred_provider)
    explicit_config = _compatible_config_from_required_env(model_override=model_override)

    if hint is None:
        return explicit_config or _groq_config(model_override=model_override) or _openai_config(
            model_override=model_override
        )
    if hint == "compatible":
        return explicit_config or _groq_config(model_override=model_override) or _openai_config(
            model_override=model_override
        )
    if hint == "groq":
        if explicit_config and explicit_config.provider == "groq":
            return explicit_config
        return _groq_config(model_override=model_override)
    if hint == "openai":
        if explicit_config and explicit_config.provider == "openai":
            return explicit_config
        return _openai_config(model_override=model_override)
    if hint == "grok":
        if explicit_config and explicit_config.provider in {"grok", "compatible"}:
            return explicit_config
        return None
    return explicit_config


def _compatible_config_from_required_env(*, model_override: str | None = None) -> LLMConfig | None:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = model_override or os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not (api_base_url and model_name and hf_token):
        return None
    return LLMConfig(
        provider=provider_label_from_base_url(api_base_url),
        base_url=api_base_url,
        model_name=model_name,
        api_key=hf_token,
    )


def _openai_config(*, model_override: str | None = None) -> LLMConfig | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return LLMConfig(
        provider="openai",
        base_url=DEFAULT_OPENAI_BASE_URL,
        model_name=model_override or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        api_key=api_key,
    )


def _groq_config(*, model_override: str | None = None) -> LLMConfig | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    return LLMConfig(
        provider="groq",
        base_url=DEFAULT_GROQ_BASE_URL,
        model_name=model_override or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
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
