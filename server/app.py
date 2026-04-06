"""FastAPI application for the Support Triage environment."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Query
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel

try:
    from ..baseline import DEFAULT_MODEL, BaselineRunResult, run_baseline_sync
    from ..graders import GraderResult, grade_support_episode
    from ..llm_client import resolve_llm_config
    from ..models import SupportTriageAction, SupportTriageObservation, SupportTriageState
    from ..tasks import public_task_cards
    from .support_triage_environment import SupportTriageEnvironment
except ImportError:
    from baseline import DEFAULT_MODEL, BaselineRunResult, run_baseline_sync
    from graders import GraderResult, grade_support_episode
    from llm_client import resolve_llm_config
    from models import SupportTriageAction, SupportTriageObservation, SupportTriageState
    from tasks import public_task_cards
    from server.support_triage_environment import SupportTriageEnvironment


class TasksResponse(BaseModel):
    """Response payload for /tasks."""

    tasks: list[dict[str, object]]
    action_schema: dict[str, object]
    reset_parameters: dict[str, object]


class RuntimeResponse(BaseModel):
    llm_configured: bool
    provider: str | None = None
    model_name: str | None = None
    api_base_url: str | None = None


app: FastAPI = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage_env",
    max_concurrent_envs=8,
)


@app.get("/", summary="Service root")
def root() -> dict[str, object]:
    return {
        "ok": True,
        "name": "support_triage_env",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }


@app.get("/runtime", response_model=RuntimeResponse, summary="Show current LLM runtime config")
def runtime() -> RuntimeResponse:
    config = resolve_llm_config()
    return RuntimeResponse(
        llm_configured=config is not None,
        provider=config.provider if config else None,
        model_name=config.model_name if config else None,
        api_base_url=config.base_url if config else None,
    )


@app.get("/tasks", response_model=TasksResponse, summary="List tasks and action schema")
def list_tasks() -> TasksResponse:
    return TasksResponse(
        tasks=public_task_cards(),
        action_schema=SupportTriageAction.model_json_schema(),
        reset_parameters={
            "task_id": {
                "type": "string",
                "description": "Optional task id passed to reset(task_id=...)",
            },
            "custom_ticket_id": {
                "type": "string",
                "description": "Optional ticket id for task_id=custom_message_sandbox",
            },
            "custom_subject": {
                "type": "string",
                "description": "Optional custom subject for task_id=custom_message_sandbox",
            },
            "custom_body": {
                "type": "string",
                "description": "Required custom customer message for task_id=custom_message_sandbox",
            },
            "custom_customer_tier": {
                "type": "string",
                "description": "Optional starter|pro|enterprise|vip override for task_id=custom_message_sandbox",
            },
            "custom_channel": {
                "type": "string",
                "description": "Optional email|chat override for task_id=custom_message_sandbox",
            },
            "custom_hours_open": {
                "type": "integer",
                "description": "Optional age in hours for task_id=custom_message_sandbox",
            },
            "custom_contains_sensitive_data": {
                "type": "boolean",
                "description": "Optional explicit sensitive-data flag for task_id=custom_message_sandbox",
            },
        },
    )


@app.post("/grader", response_model=GraderResult, summary="Grade a finished episode state")
def grade_episode(state: SupportTriageState) -> GraderResult:
    try:
        return grade_support_episode(state)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/baseline", response_model=BaselineRunResult, summary="Run the baseline policy")
async def baseline(
    agent: str = Query("auto", pattern="^(auto|groq|grok|openai|compatible|scripted)$"),
    model: str = Query(DEFAULT_MODEL),
) -> BaselineRunResult:
    return await asyncio.to_thread(
        run_baseline_sync,
        model=model,
        agent_backend=agent,
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
