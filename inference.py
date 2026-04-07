"""Root-level submission inference script using the OpenAI client."""

from __future__ import annotations

import os
import sys

from baseline import scripted_policy_action
from graders import grade_support_episode
from llm_client import resolve_openai_config
from models import SupportTriageAction, SupportTriageObservation
from openai_reply import generate_llm_reply
from server.support_triage_environment import SupportTriageEnvironment
from tasks import TASKS

BENCHMARK = os.getenv("SUPPORT_TRIAGE_BENCHMARK", "support_triage_env")


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_decimal(value: float) -> str:
    return f"{value:.2f}"


def _format_error(value: str | None) -> str:
    if not value:
        return "null"
    return value.replace("\n", " ").strip() or "null"


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _format_action(action: SupportTriageAction) -> str:
    if action.action_type == "complete_episode":
        return "complete_episode()"
    if action.action_type == "open_ticket":
        return f"open_ticket('{_quote(action.ticket_id or '')}')"
    if action.action_type == "reply_to_ticket":
        return f"reply_to_ticket('{_quote(action.ticket_id or '')}')"
    if action.action_type == "add_internal_note":
        return f"add_internal_note('{_quote(action.ticket_id or '')}')"
    if action.action_type == "redact_sensitive_data":
        return f"redact_sensitive_data('{_quote(action.ticket_id or '')}')"

    updates: list[str] = []
    if action.priority:
        updates.append(f"priority='{_quote(action.priority)}'")
    if action.queue:
        updates.append(f"queue='{_quote(action.queue)}'")
    if action.status:
        updates.append(f"status='{_quote(action.status)}'")
    if action.tags:
        tags = "[" + ",".join(f"'{_quote(tag)}'" for tag in action.tags) + "]"
        updates.append(f"tags={tags}")
    joined = ", ".join(updates)
    suffix = f", {joined}" if joined else ""
    return f"update_ticket('{_quote(action.ticket_id or '')}'{suffix})"


def emit_start(*, task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)


def emit_step(
    *,
    step: int,
    action: SupportTriageAction,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    print(
        "[STEP] "
        f"step={step} "
        f"action={_format_action(action)} "
        f"reward={_format_decimal(reward)} "
        f"done={_format_bool(done)} "
        f"error={_format_error(error)}",
        flush=True,
    )


def emit_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_values = ",".join(_format_decimal(value) for value in rewards)
    print(
        "[END] "
        f"success={_format_bool(success)} "
        f"steps={steps} "
        f"score={_format_decimal(score)} "
        f"rewards={reward_values}",
        flush=True,
    )


def choose_action(
    observation: SupportTriageObservation,
    model_name: str,
) -> tuple[SupportTriageAction, str | None]:
    action = scripted_policy_action(observation)
    if action.action_type != "reply_to_ticket" or not action.ticket_id:
        return action, None
    try:
        message, provider = generate_llm_reply(
            observation=observation,
            ticket_id=action.ticket_id,
            model_override=model_name,
        )
    except Exception:
        return action, "scripted_fallback"
    return (
        SupportTriageAction(
            action_type="reply_to_ticket",
            ticket_id=action.ticket_id,
            message=message,
        ),
        provider,
    )


def run_task(task_id: str, model_name: str) -> dict[str, float | int | bool | str]:
    env = SupportTriageEnvironment()
    rewards: list[float] = []
    success = False
    score = 0.0
    steps = 0
    emit_start(task_name=task_id, model_name=model_name)

    try:
        observation = env.reset(task_id=task_id)
        while not observation.done and env.state.step_count < env.state.max_steps:
            action, _provider_used = choose_action(observation, model_name)
            observation = env.step(action)
            rewards.append(observation.reward)
            latest = env.state.action_history[-1] if env.state.action_history else None
            error = env.state.latest_feedback if latest and latest.invalid else None
            emit_step(
                step=env.state.step_count,
                action=action,
                reward=observation.reward,
                done=observation.done,
                error=error,
            )

        grade = grade_support_episode(env.state)
        success = grade.passed
        score = grade.score
        steps = env.state.step_count
        return {
            "task_id": task_id,
            "score": grade.score,
            "passed": grade.passed,
            "steps": env.state.step_count,
            "cumulative_reward": env.state.cumulative_reward,
        }
    except Exception as exc:
        steps = env.state.step_count
        print(f"ERROR: task={task_id} detail={exc}", file=sys.stderr, flush=True)
        return {
            "task_id": task_id,
            "score": 0.0,
            "passed": False,
            "steps": steps,
            "cumulative_reward": env.state.cumulative_reward,
        }
    finally:
        emit_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    config = resolve_openai_config()
    if config is None:
        print(
            "ERROR: Set API_BASE_URL, MODEL_NAME, and HF_TOKEN for OpenAI.",
            file=sys.stderr,
        )
        sys.exit(1)

    results: list[dict[str, float | int | bool | str]] = []
    for task in TASKS:
        results.append(run_task(task.task_id, config.model_name))

    failed = [result for result in results if not result["passed"]]
    if failed:
        print(
            "WARNING: one or more tasks did not pass. "
            + ", ".join(str(result["task_id"]) for result in failed),
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
