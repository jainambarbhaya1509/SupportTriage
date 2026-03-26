---
title: Support Triage OpenEnv
emoji: 📥
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8501
tags:
  - openenv
  - rl
  - evaluation
  - customer-support
  - streamlit
---

# Support Triage OpenEnv

`support_triage_env` is a real-world OpenEnv environment for customer-support operations. The agent works a realistic inbox with billing, security, and product tickets, choosing how to inspect, route, redact, document, and reply to each case.

This environment is designed for RL-style agent learning rather than one-shot classification. The reward is dense, deterministic, and shaped by incremental progress toward the hidden target workflow for each task.

For a file-by-file architecture explanation and full runtime walkthrough, see [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md).

## Why This Is Real-World

Humans do this work every day in SaaS, fintech, and enterprise support teams:

- prioritizing urgent security incidents over routine queue work
- routing issues to the correct specialty queue
- redacting sensitive customer data before responding
- resolving low-risk requests while escalating high-risk ones
- writing safe, policy-aligned customer replies and internal notes

That combination makes the environment useful for evaluating long-horizon operational agents, not just single-turn labeling models.

## OpenEnv Interface

- `reset(task_id=...) -> SupportTriageObservation`
- `step(SupportTriageAction) -> SupportTriageObservation`
- `state() -> SupportTriageState`
- `openenv.yaml` included at repo root

The server is exposed through `server.app:app` and is compatible with `openenv validate`.

## Action Space

The action model is `SupportTriageAction`.

- `open_ticket`
  Opens a ticket so the agent can inspect the full body before making changes.
- `update_ticket`
  Sets any combination of `priority`, `queue`, `status`, and `tags`.
- `reply_to_ticket`
  Sends a customer-visible message. If `message` is omitted and `GROQ_API_KEY` is configured on the server, the backend generates the reply with Groq.
- `add_internal_note`
  Adds an internal note for downstream human responders.
- `redact_sensitive_data`
  Removes sensitive data from the visible conversation before a reply.
- `complete_episode`
  Ends the episode when the agent believes the queue is handled.

## Observation Space

The observation model is `SupportTriageObservation`.

- task metadata: `task_id`, `task_title`, `difficulty`, `objective`, `success_criteria`
- inbox summary: all visible tickets with current routing state
- active ticket: full body, latest reply, latest internal note
- learning signals: `reward`, `completion_score`, `remaining_steps`, `invalid_action_count`
- feedback: a natural-language description of what the last action did

The full internal state is available through `SupportTriageState` and the `state()` API.

## Tasks

Three deterministic tasks ship with the environment:

1. `billing_refund_easy`
   Single-ticket billing escalation. The agent must inspect, classify, tag, and acknowledge a refund complaint safely.
2. `security_and_invoice_medium`
   Mixed queue with one urgent account-compromise ticket and one routine invoice request. The grader checks urgency ordering and correct routing.
3. `vip_incident_hard`
   Mixed VIP queue with a leaked API key, raw card data in a refund request, and a low-priority product request. The grader checks redaction-before-reply, internal notes, and correct multi-ticket workflow.

For manual testing, there is also `custom_message_sandbox`, which lets you create a one-off ticket by passing `custom_subject` and `custom_body` to `reset(...)`.

## Reward Design

The reward is based on incremental grader progress:

- positive reward when the agent opens the right tickets and moves fields toward the hidden target workflow
- partial reward for getting some fields right even before the episode is perfect
- penalties for invalid actions, repeated no-ops, premature replies on sensitive tickets, and wasting steps

The final grader score is always normalized to `0.0` to `1.0`.

## Additional Endpoints

- `GET /tasks`
  Lists all tasks plus the action JSON schema.
- `POST /grader`
  Grades a serialized `SupportTriageState` and returns a deterministic `0.0-1.0` score.
- `GET /baseline`
  Runs the baseline agent across all three tasks and returns aggregate results.

`/grader` expects the same JSON returned by `state()`.

## Baselines

Two baselines are included:

- Groq baseline
  `baseline.py` reads `GROQ_API_KEY`, keeps the workflow policy deterministic, and uses Groq-generated text for customer replies through Groq's OpenAI-compatible chat completions API.
- Scripted fallback baseline
  Deterministic reference policy used for smoke tests and Spaces where an API key is not configured.

Scripted baseline scores:

- easy: `1.00`
- medium: `1.00`
- hard: `1.00`
- mean: `1.00`

Groq baseline command:

```bash
python baseline.py --agent groq --model llama-3.1-8b-instant
```

If `GROQ_API_KEY` is missing, `--agent auto` falls back to the scripted baseline so `/baseline` still works in a default deployment.

For manual WebSocket or Postman testing, you can also omit `message` on `reply_to_ticket` and let the backend generate the customer reply from Groq.

If your local virtual environment is out of date, reinstall project dependencies:

```bash
pip install -e ".[dev]"
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Run the separate Streamlit console:

```bash
pip install -e ".[ui]"
streamlit run streamlit_app.py
```

The Streamlit app reuses the same environment and grader directly in-process, so judged tasks, custom sandbox tickets, grading, and Groq-backed blank replies behave the same way as the API server.

Custom sandbox reset example over WebSocket:

```json
{
  "type": "reset",
  "data": {
    "task_id": "custom_message_sandbox",
    "custom_ticket_id": "CUSTOM-4242",
    "custom_subject": "Need help with my refund",
    "custom_body": "Hi team, I was charged twice and need help understanding the refund status.",
    "custom_customer_tier": "pro",
    "custom_channel": "email"
  }
}
```

Validate locally:

```bash
openenv validate
```

Run tests:

```bash
pytest
```

## Docker

Build and run:

```bash
docker build -t support-triage-env .
docker run --rm -p 8000:8000 support-triage-env
```

The repo also includes `server/Dockerfile` for OpenEnv-style builds.

## Hugging Face Spaces

The repository is ready for a Docker Space that launches the separate Streamlit console:

- root `README.md` includes Docker Space metadata with `app_port: 8501`
- root `Dockerfile` installs the `ui` extra and starts `streamlit_app.py`
- blank auto-replies use `GROQ_API_KEY` from Space secrets

For local development, the FastAPI/OpenEnv server still exists at `server.app:app`, but the Hugging Face Space entry point is the Streamlit UI.

## Project Structure

```text
.
├── README.md
├── Dockerfile
├── openenv.yaml
├── models.py
├── tasks.py
├── graders.py
├── client.py
├── baseline.py
├── server/
│   ├── app.py
│   ├── Dockerfile
│   └── support_triage_environment.py
└── tests/
    └── test_environment.py
```
