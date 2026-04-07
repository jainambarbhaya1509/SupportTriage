---
title: Support Triage OpenEnv
emoji: 📥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - rl
  - evaluation
  - customer-support
  - fastapi
---

# Support Triage OpenEnv

`support_triage_env` is a real-world OpenEnv environment for customer-support operations. Agents triage billing, security, and product tickets by opening messages, routing work, applying redaction, writing notes, and sending safe customer replies.

This repo is now OpenAI-only.

## OpenEnv Interface

- `reset(task_id=...) -> SupportTriageObservation`
- `step(SupportTriageAction) -> SupportTriageObservation`
- `state() -> SupportTriageState`
- `openenv.yaml` points to `server.app:app`

The API server exposes standard OpenEnv endpoints plus:

- `GET /`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`
- `GET /runtime`

## Tasks

1. `billing_refund_easy`
2. `security_and_invoice_medium`
3. `vip_incident_hard`

A manual `custom_message_sandbox` task is also included for ad-hoc testing.

## Action Space

`SupportTriageAction` supports:

- `open_ticket`
- `update_ticket`
- `reply_to_ticket`
- `add_internal_note`
- `redact_sensitive_data`
- `complete_episode`

If `reply_to_ticket.message` is omitted, the backend can auto-generate the reply with OpenAI.

## Reward Design

Rewards are shaped by grader progress over the full trajectory. Agents earn partial credit for correct routing and workflow progress, and receive penalties for invalid or unsafe behavior.

## OpenAI Configuration

The required variables are:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-mini
HF_TOKEN=your_openai_api_key
```

All model calls in `inference.py`, the environment, and the baseline use the OpenAI Python client.

Optional local fallback:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Baseline and Inference

Two root scripts are provided:

- `baseline.py`
  Reproducible evaluator with `scripted`, `openai`, and `auto` modes.
- `inference.py`
  Submission script that emits structured `[START]`, `[STEP]`, and `[END]` logs.

Examples:

```bash
python3 baseline.py --agent scripted
python3 baseline.py --agent openai --model gpt-4.1-mini
python3 inference.py
```

Scripted baseline scores:

- easy: `1.00`
- medium: `1.00`
- hard: `1.00`
- mean: `1.00`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the OpenEnv server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Validate locally:

```bash
openenv validate
pytest
```

## Streamlit Utilities

The repo also keeps the separate Streamlit operator console:

```bash
pip install -e ".[ui]"
streamlit run streamlit_app.py
```

That UI is for local operations and demos. The root Docker and HF Space deployment is API-first for the OpenEnv validator.

## Docker

```bash
docker build -t support-triage-env .
docker run --rm -p 8000:8000 support-triage-env
```

## Hugging Face Spaces

This repository is configured for a Docker Space that serves the OpenEnv API.

Use these Space secrets and variables:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-mini
HF_TOKEN=your_openai_api_key
```

## Project Structure

```text
.
├── README.md
├── Dockerfile
├── openenv.yaml
├── inference.py
├── baseline.py
├── llm_client.py
├── openai_reply.py
├── models.py
├── tasks.py
├── graders.py
├── client.py
├── streamlit_app.py
├── server/
│   ├── app.py
│   ├── Dockerfile
│   └── support_triage_environment.py
└── tests/
    └── test_environment.py
```
