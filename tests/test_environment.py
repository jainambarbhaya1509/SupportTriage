"""Deterministic smoke tests for the support triage environment."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server.support_triage_environment as env_module
from baseline import run_baseline_sync, scripted_policy_action
from graders import grade_support_episode
from groq_reply import build_reply_prompt
from llm_client import resolve_llm_config
from models import SupportTriageAction
from server.app import app
from server.support_triage_environment import SupportTriageEnvironment
from tasks import CUSTOM_TASK_ID, TASKS, get_task


def test_scripted_policy_solves_all_tasks() -> None:
    env = SupportTriageEnvironment()

    for task in TASKS:
        observation = env.reset(task_id=task.task_id)
        while not observation.done:
            observation = env.step(scripted_policy_action(observation))

        result = grade_support_episode(env.state)
        assert result.score == 1.0, task.task_id


def test_invalid_update_without_opening_ticket_is_penalized() -> None:
    env = SupportTriageEnvironment()
    env.reset(task_id="billing_refund_easy")

    observation = env.step(
        SupportTriageAction(
            action_type="update_ticket",
            ticket_id="BILL-1001",
            priority="high",
        )
    )

    assert observation.reward < 0
    assert env.state.invalid_action_count == 1


def test_reply_before_redaction_hurts_hard_task_score() -> None:
    env = SupportTriageEnvironment()
    env.reset(task_id="vip_incident_hard")
    env.step(SupportTriageAction(action_type="open_ticket", ticket_id="BILL-3302"))
    env.step(
        SupportTriageAction(
            action_type="reply_to_ticket",
            ticket_id="BILL-3302",
            message="We will refund card 4242 4242 4242 4242 right away.",
        )
    )

    graded = grade_support_episode(env.state)
    assert graded.breakdown["workflow:redact_before_reply"] == 0.0
    assert graded.score < 0.4


def test_reply_without_message_uses_auto_llm_when_configured(monkeypatch) -> None:
    env = SupportTriageEnvironment()
    env.reset(task_id="billing_refund_easy")
    env.step(SupportTriageAction(action_type="open_ticket", ticket_id="BILL-1001"))

    monkeypatch.setattr(
        env_module,
        "generate_llm_reply",
        lambda observation, ticket_id, preferred_provider=None, model_override=None: (
            "I have escalated this billing refund request for review.",
            "groq",
        ),
    )

    observation = env.step(
        SupportTriageAction(
            action_type="reply_to_ticket",
            ticket_id="BILL-1001",
        )
    )

    assert observation.action_feedback == "Sent groq-generated reply on BILL-1001."
    assert env.state.active_ticket().public_reply == "I have escalated this billing refund request for review."


def test_custom_message_reset_creates_ticket() -> None:
    env = SupportTriageEnvironment()

    observation = env.reset(
        task_id=CUSTOM_TASK_ID,
        custom_ticket_id="CUSTOM-4242",
        custom_subject="My custom support issue",
        custom_body="Hello team, please help me change the invoice recipient on my account.",
        custom_customer_tier="pro",
        custom_channel="email",
    )

    assert observation.task_id == CUSTOM_TASK_ID
    assert observation.inbox[0].ticket_id == "CUSTOM-4242"
    assert observation.inbox[0].subject == "My custom support issue"
    assert observation.active_ticket is None


def test_custom_task_id_is_registered() -> None:
    task = get_task(CUSTOM_TASK_ID)

    assert task.task_id == CUSTOM_TASK_ID
    assert task.max_steps == 10


def test_custom_ticket_prompt_builds_without_hidden_rubric() -> None:
    env = SupportTriageEnvironment()
    observation = env.reset(
        task_id=CUSTOM_TASK_ID,
        custom_ticket_id="CUSTOM-4242",
        custom_subject="Need refund help",
        custom_body="Hi team, I was charged twice and need help with the refund.",
    )
    observation = env.step(
        SupportTriageAction(action_type="open_ticket", ticket_id="CUSTOM-4242")
    )

    prompt = build_reply_prompt(observation, "CUSTOM-4242")

    assert "CUSTOM-4242" in prompt
    assert "Required reply concepts: none" in prompt


def test_auto_baseline_falls_back_to_scripted_without_llm_config(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = run_baseline_sync(agent_backend="auto")

    assert result.agent_backend == "scripted"
    assert result.mean_score == 1.0


def test_required_env_resolves_compatible_provider(monkeypatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("MODEL_NAME", "llama-3.1-8b-instant")
    monkeypatch.setenv("HF_TOKEN", "test-key")

    config = resolve_llm_config()

    assert config is not None
    assert config.provider == "groq"
    assert config.model_name == "llama-3.1-8b-instant"


def test_root_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_streamlit_app_source_compiles() -> None:
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "streamlit_app.py"),
        doraise=True,
    )


def test_mail_bridge_source_compiles() -> None:
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "mail_bridge.py"),
        doraise=True,
    )


def test_inference_source_compiles() -> None:
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "inference.py"),
        doraise=True,
    )
