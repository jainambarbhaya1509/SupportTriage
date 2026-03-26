"""Groq-backed reply generation helpers shared by the baseline and environment."""

from __future__ import annotations

import os
import re
from typing import Any

import requests

try:
    from .models import SupportTriageObservation
    from .tasks import CUSTOM_TASK_ID, get_task, infer_sensitive_content
except ImportError:
    from models import SupportTriageObservation
    from tasks import CUSTOM_TASK_ID, get_task, infer_sensitive_content

DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TIMEOUT_S = 30.0
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DIGIT_RE = re.compile(r"\d")
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.DOTALL)


def clean_groq_text(raw: Any) -> str:
    """Normalize Groq output into a plain single-paragraph reply."""

    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                part = item.get("text") or item.get("content") or ""
            else:
                part = getattr(item, "text", None) or getattr(item, "content", "")
            if part:
                parts.append(str(part))
        text = "\n".join(parts)
    else:
        text = str(raw or "")

    text = CODE_FENCE_RE.sub("", text).strip()
    text = text.strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_reply_prompt(
    observation: SupportTriageObservation,
    ticket_id: str,
) -> str:
    """Build a safe, task-aware prompt for customer reply generation."""

    task = get_task(observation.task_id)
    active_ticket = observation.active_ticket
    if active_ticket is None or active_ticket.ticket_id != ticket_id:
        raise ValueError(f"Ticket {ticket_id} must be active before generating a reply.")

    expected = task.expected.get(ticket_id)
    required_keywords = (
        ", ".join(expected.required_reply_keywords) if expected else "none"
    ) or "none"
    forbidden_keywords = (
        ", ".join(expected.forbidden_reply_keywords) if expected else "none"
    ) or "none"
    extra_rules: list[str] = [
        "Keep the reply concise and professional.",
        "Acknowledge the issue and explain the next step.",
        "Do not promise outcomes you cannot guarantee.",
        "Return plain text only with no markdown or bullet points.",
    ]
    contains_sensitive_content = infer_sensitive_content(active_ticket.body)
    if expected and (expected.redaction_required or expected.forbid_digit_echo):
        extra_rules.append("Do not repeat any raw payment card numbers, secrets, or digits.")
    elif observation.task_id == CUSTOM_TASK_ID and contains_sensitive_content:
        extra_rules.append("Do not repeat any raw payment card numbers, secrets, or digits.")
    if observation.task_id == "vip_incident_hard":
        extra_rules.append("Avoid digits entirely in the reply.")
    elif observation.task_id == CUSTOM_TASK_ID and contains_sensitive_content:
        extra_rules.append("Avoid repeating sensitive values from the customer message.")

    return (
        "You are drafting a single customer-support reply.\n"
        f"Task objective: {observation.objective}\n"
        f"Ticket ID: {active_ticket.ticket_id}\n"
        f"Customer tier: {active_ticket.customer_tier}\n"
        f"Channel: {active_ticket.channel}\n"
        f"Subject: {active_ticket.subject}\n"
        f"Customer message: {active_ticket.body}\n"
        f"Required reply concepts: {required_keywords}\n"
        f"Forbidden reply concepts: {forbidden_keywords}\n"
        "Rules:\n"
        + "\n".join(f"- {rule}" for rule in extra_rules)
    )


def generate_groq_reply(
    api_key: str,
    model: str,
    observation: SupportTriageObservation,
    ticket_id: str,
) -> str:
    """Generate a customer-safe reply from Groq and sanitize the result."""

    prompt = build_reply_prompt(observation=observation, ticket_id=ticket_id)
    response = requests.post(
        GROQ_CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_completion_tokens": 180,
        },
        timeout=DEFAULT_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    raw_text = ""
    if choices and isinstance(choices[0], dict):
        raw_text = choices[0].get("message", {}).get("content", "")

    cleaned = clean_groq_text(raw_text)
    if observation.task_id == "vip_incident_hard":
        cleaned = DIGIT_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
