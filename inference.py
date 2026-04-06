"""Root-level submission inference script using the OpenAI client."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from baseline import scripted_policy_action
from graders import grade_support_episode
from groq_reply import generate_llm_reply
from llm_client import resolve_llm_config
from models import SupportTriageAction, SupportTriageObservation
from server.support_triage_environment import SupportTriageEnvironment
from tasks import TASKS


def emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}", flush=True)


def choose_action(
    observation: SupportTriageObservation,
    provider_hint: str | None,
    model_name: str,
) -> tuple[SupportTriageAction, str | None]:
    action = scripted_policy_action(observation)
    if action.action_type != "reply_to_ticket" or not action.ticket_id:
        return action, None
    message, provider = generate_llm_reply(
        observation=observation,
        ticket_id=action.ticket_id,
        preferred_provider=provider_hint,
        model_override=model_name,
    )
    return (
        SupportTriageAction(
            action_type="reply_to_ticket",
            ticket_id=action.ticket_id,
            message=message,
        ),
        provider,
    )


def run_task(task_id: str, provider_hint: str | None, model_name: str) -> dict[str, Any]:
    env = SupportTriageEnvironment()
    observation = env.reset(task_id=task_id)
    emit(
        "STEP",
        {
            "event": "task_start",
            "task_id": task_id,
            "task_title": observation.task_title,
            "difficulty": observation.difficulty,
            "objective": observation.objective,
            "max_steps": env.state.max_steps,
        },
    )

    while not observation.done and env.state.step_count < env.state.max_steps:
        action, provider_used = choose_action(observation, provider_hint, model_name)
        observation = env.step(action)
        emit(
            "STEP",
            {
                "event": "action",
                "task_id": task_id,
                "step": env.state.step_count,
                "action_type": action.action_type,
                "ticket_id": action.ticket_id,
                "provider": provider_used,
                "reward": observation.reward,
                "cumulative_reward": observation.cumulative_reward,
                "completion_score": observation.completion_score,
                "done": observation.done,
            },
        )

    grade = grade_support_episode(env.state)
    payload = {
        "event": "task_end",
        "task_id": task_id,
        "score": grade.score,
        "passed": grade.passed,
        "steps": env.state.step_count,
        "cumulative_reward": env.state.cumulative_reward,
    }
    emit("STEP", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the submission inference script.")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=("auto", "groq", "grok", "openai", "compatible"),
        help="Optional provider hint. The required env vars still control the endpoint.",
    )
    args = parser.parse_args()

    provider_hint = None if args.provider == "auto" else args.provider
    config = resolve_llm_config(provider_hint)
    if config is None:
        print(
            "ERROR: Set API_BASE_URL, MODEL_NAME, and HF_TOKEN for an OpenAI-compatible endpoint.",
            file=sys.stderr,
        )
        sys.exit(1)

    emit(
        "START",
        {
            "provider": config.provider,
            "model_name": config.model_name,
            "api_base_url": config.base_url,
            "task_ids": [task.task_id for task in TASKS],
        },
    )

    results: list[dict[str, Any]] = []
    for task in TASKS:
        results.append(run_task(task.task_id, config.provider, config.model_name))

    mean_score = round(sum(item["score"] for item in results) / len(results), 4)
    emit(
        "END",
        {
            "provider": config.provider,
            "model_name": config.model_name,
            "api_base_url": config.base_url,
            "mean_score": mean_score,
            "task_results": results,
            "task_count": len(results),
        },
    )


if __name__ == "__main__":
    main()
