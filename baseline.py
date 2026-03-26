"""Baseline runners for the support triage environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

try:
    from .client import SupportTriageEnv
    from .graders import grade_support_episode
    from .groq_reply import DEFAULT_GROQ_MODEL, generate_groq_reply
    from .models import SupportTriageAction, SupportTriageObservation
    from .server.support_triage_environment import SupportTriageEnvironment
    from .tasks import TASKS, get_task
except ImportError:
    from client import SupportTriageEnv
    from graders import grade_support_episode
    from groq_reply import DEFAULT_GROQ_MODEL, generate_groq_reply
    from models import SupportTriageAction, SupportTriageObservation
    from server.support_triage_environment import SupportTriageEnvironment
    from tasks import TASKS, get_task

import requests

DEFAULT_MODEL = DEFAULT_GROQ_MODEL
DEFAULT_TIMEOUT_S = 30.0


class BaselineTaskResult(BaseModel):
    """Per-task baseline result."""

    task_id: str
    title: str
    difficulty: str
    agent_backend: str
    score: float
    steps_taken: int
    cumulative_reward: float


class BaselineRunResult(BaseModel):
    """Aggregate result for /baseline and CLI usage."""

    model: str | None = None
    agent_backend: str
    mean_score: float
    task_results: list[BaselineTaskResult] = Field(default_factory=list)


def _reference_reply(ticket_id: str) -> str:
    replies = {
        "BILL-1001": "Sorry about the duplicate charge. I am escalating this to billing so they can review the refund.",
        "SEC-2201": "We are treating this as a security issue, the account lock is in progress, and this has been escalated immediately.",
        "BILL-2202": "You can download the invoice from the billing portal, so I am marking this invoice request resolved.",
        "SEC-3301": "Please rotate the exposed key now. Security has been escalated and we are treating this as urgent.",
        "BILL-3302": "For your secure billing review, I have escalated this refund request to the billing team.",
        "PROD-3303": "Thanks for the feedback. I have triaged this for the product roadmap.",
    }
    return replies[ticket_id]


def _reference_note(ticket_id: str) -> str | None:
    notes = {
        "SEC-3301": "Request customer revoke active sessions, rotate the key, and start an audit of recent usage.",
    }
    return notes.get(ticket_id)


def _ordered_ticket_ids(task_id: str) -> list[str]:
    task = get_task(task_id)
    ordered = []
    if task.first_decision_ticket_id:
        ordered.append(task.first_decision_ticket_id)
    for ticket in task.tickets:
        if ticket.ticket_id not in ordered:
            ordered.append(ticket.ticket_id)
    return ordered


def scripted_policy_action(observation: SupportTriageObservation) -> SupportTriageAction:
    task = get_task(observation.task_id)
    ordered_ticket_ids = _ordered_ticket_ids(task.task_id)
    inbox_map = {ticket.ticket_id: ticket for ticket in observation.inbox}

    for ticket_id in ordered_ticket_ids:
        target = inbox_map[ticket_id]
        expected = task.expected[ticket_id]
        is_active = observation.active_ticket is not None and observation.active_ticket.ticket_id == ticket_id
        note_complete = target.internal_note_present or bool(
            observation.active_ticket and observation.active_ticket.ticket_id == ticket_id and observation.active_ticket.internal_note
        )
        tags_complete = set(target.tags) == set(expected.required_tags)
        work_is_complete = (
            target.opened
            and target.priority == expected.priority
            and target.queue == expected.queue
            and target.status == expected.status
            and tags_complete
            and (not expected.redaction_required or target.redaction_applied)
            and (not expected.required_reply_keywords or target.public_reply_sent)
            and (not expected.required_note_keywords or note_complete)
        )

        if work_is_complete:
            continue

        if not target.opened:
            return SupportTriageAction(action_type="open_ticket", ticket_id=ticket_id)

        if not is_active:
            return SupportTriageAction(action_type="open_ticket", ticket_id=ticket_id)

        if expected.redaction_required and not target.redaction_applied:
            return SupportTriageAction(
                action_type="redact_sensitive_data",
                ticket_id=ticket_id,
                reason="Remove sensitive customer data before any reply is sent.",
            )

        note = _reference_note(ticket_id)
        if note and not note_complete:
            return SupportTriageAction(
                action_type="add_internal_note",
                ticket_id=ticket_id,
                note=note,
            )

        if (
            target.priority != expected.priority
            or target.queue != expected.queue
            or target.status != expected.status
            or set(target.tags) != set(expected.required_tags)
        ):
            return SupportTriageAction(
                action_type="update_ticket",
                ticket_id=ticket_id,
                priority=expected.priority,
                queue=expected.queue,
                status=expected.status,
                tags=list(expected.required_tags),
            )

        reply = _reference_reply(ticket_id)
        if not target.public_reply_sent and expected.required_reply_keywords:
            return SupportTriageAction(
                action_type="reply_to_ticket",
                ticket_id=ticket_id,
                message=reply,
            )

    return SupportTriageAction(
        action_type="complete_episode",
        summary=f"Completed task {task.task_id} with scripted policy.",
    )


def groq_policy_action(
    observation: SupportTriageObservation,
    *,
    api_key: str,
    model: str,
) -> SupportTriageAction:
    action = scripted_policy_action(observation)
    if action.action_type != "reply_to_ticket" or not action.ticket_id:
        return action

    return SupportTriageAction(
        action_type="reply_to_ticket",
        ticket_id=action.ticket_id,
        message=generate_groq_reply(
            api_key=api_key,
            model=model,
            observation=observation,
            ticket_id=action.ticket_id,
        )
        or _reference_reply(action.ticket_id),
    )


def _choose_action(
    observation: SupportTriageObservation,
    agent_backend: str,
    model: str,
    api_key: str | None,
) -> SupportTriageAction:
    if agent_backend == "scripted":
        return scripted_policy_action(observation)
    if not api_key:
        raise RuntimeError(
            "Groq agent requested but GROQ_API_KEY is not set"
        )
    return groq_policy_action(observation, api_key=api_key, model=model)


def _wait_for_server(base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = requests.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.25)
            continue
        time.sleep(0.25)
    raise TimeoutError(f"Server at {base_url} did not become ready in time")


@contextmanager
def _local_server(base_url: str) -> Iterator[str]:
    repo_root = Path(__file__).resolve().parent
    port = base_url.rsplit(":", 1)[-1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server(base_url)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_baseline_sync(
    base_url: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    agent_backend: str = "auto",
) -> BaselineRunResult:
    """Run the baseline against all tasks and return reproducible scores."""

    api_key = os.environ.get("GROQ_API_KEY")
    resolved_backend = agent_backend
    if agent_backend == "auto":
        resolved_backend = "groq" if api_key else "scripted"

    if resolved_backend == "groq" and not api_key:
        raise RuntimeError("GROQ_API_KEY is required when --agent groq is used")

    def _run_with_client(resolved_base_url: str) -> BaselineRunResult:
        task_results: list[BaselineTaskResult] = []
        with SupportTriageEnv(base_url=resolved_base_url).sync() as env:
            for task in TASKS:
                result = env.reset(task_id=task.task_id)
                while not result.done:
                    action = _choose_action(
                        result.observation,
                        agent_backend=resolved_backend,
                        model=model,
                        api_key=api_key,
                    )
                    result = env.step(action)
                state = env.state()
                grade = grade_support_episode(state)
                task_results.append(
                    BaselineTaskResult(
                        task_id=task.task_id,
                        title=task.title,
                        difficulty=task.difficulty,
                        agent_backend=resolved_backend,
                        score=grade.score,
                        steps_taken=state.step_count,
                        cumulative_reward=round(state.cumulative_reward, 4),
                    )
                )

        mean_score = round(
            sum(result.score for result in task_results) / len(task_results), 4
        )
        return BaselineRunResult(
            model=model if resolved_backend == "groq" else None,
            agent_backend=resolved_backend,
            mean_score=mean_score,
            task_results=task_results,
        )

    def _run_direct() -> BaselineRunResult:
        task_results: list[BaselineTaskResult] = []
        env = SupportTriageEnvironment()
        for task in TASKS:
            observation = env.reset(task_id=task.task_id)
            while not observation.done:
                action = _choose_action(
                    observation,
                    agent_backend=resolved_backend,
                    model=model,
                    api_key=api_key,
                )
                observation = env.step(action)
            state = env.state
            grade = grade_support_episode(state)
            task_results.append(
                BaselineTaskResult(
                    task_id=task.task_id,
                    title=task.title,
                    difficulty=task.difficulty,
                    agent_backend=resolved_backend,
                    score=grade.score,
                    steps_taken=state.step_count,
                    cumulative_reward=round(state.cumulative_reward, 4),
                )
            )

        mean_score = round(
            sum(result.score for result in task_results) / len(task_results), 4
        )
        return BaselineRunResult(
            model=model if resolved_backend == "groq" else None,
            agent_backend=resolved_backend,
            mean_score=mean_score,
            task_results=task_results,
        )

    if base_url:
        return _run_with_client(base_url.rstrip("/"))

    return _run_direct()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the support triage baseline.")
    parser.add_argument("--base-url", default=None, help="Existing environment base URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Groq model to use when --agent groq or --agent auto with a key present",
    )
    parser.add_argument(
        "--agent",
        default="auto",
        choices=("auto", "groq", "scripted"),
        help="Which baseline backend to run",
    )
    args = parser.parse_args()
    result = run_baseline_sync(
        base_url=args.base_url,
        model=args.model,
        agent_backend=args.agent,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
