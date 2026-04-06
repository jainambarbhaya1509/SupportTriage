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

The project is submission-ready for the OpenEnv Round 1 checklist:

- real-world task simulation
- full OpenEnv API via `reset()` / `step()` / `state()`
- 3 graded tasks from easy to hard
- dense reward shaping with partial progress
- root-level `inference.py`
- working `openenv.yaml`
- Docker deployment for Hugging Face Spaces
- baseline support for OpenAI, Groq, and any OpenAI-compatible endpoint

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
   Inspect, classify, tag, and acknowledge a billing refund complaint.
2. `security_and_invoice_medium`
   Prioritize an account-compromise ticket ahead of a routine invoice request.
3. `vip_incident_hard`
   Handle a leaked API key, redact a billing-risk ticket before replying, and triage a low-priority product request.

A manual `custom_message_sandbox` task is also included for ad-hoc testing.

## Action Space

`SupportTriageAction` supports:

- `open_ticket`
- `update_ticket`
- `reply_to_ticket`
- `add_internal_note`
- `redact_sensitive_data`
- `complete_episode`

If `reply_to_ticket.message` is omitted, the backend can auto-generate the reply from the configured OpenAI-compatible provider.

## Observation and Reward

Observations include:

- task metadata
- inbox summaries
- one active ticket with full body
- current reward, completion score, remaining steps, and invalid action count

Rewards are shaped by grader progress over the full trajectory. Agents earn partial credit for correct routing and workflow progress, and receive penalties for invalid or unsafe behavior.

## LLM Configuration

The hackathon validator expects these variables for inference:

```bash
API_BASE_URL=...
MODEL_NAME=...
HF_TOKEN=...
```

All model calls in `inference.py` use the OpenAI Python client against that endpoint.

### OpenAI example

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=your_openai_api_key
```

### Groq example

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=your_groq_api_key
```

### xAI Grok example

```bash
export API_BASE_URL=https://api.x.ai/v1
export MODEL_NAME=grok-3-mini-beta
export HF_TOKEN=your_xai_api_key
```

For local convenience, the code also supports fallback keys:

- `OPENAI_API_KEY`
- `GROQ_API_KEY`

But the required submission path is `API_BASE_URL + MODEL_NAME + HF_TOKEN`.

## Baseline and Inference

Two root scripts are provided:

- `baseline.py`
  Reproducible evaluator with `scripted`, `openai`, `groq`, `grok`, `compatible`, and `auto` modes.
- `inference.py`
  Submission script that emits structured `[START]`, `[STEP]`, and `[END]` logs.

Examples:

```bash
python baseline.py --agent scripted
python baseline.py --agent groq --model llama-3.1-8b-instant
python baseline.py --agent openai --model gpt-4.1-mini
python inference.py
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

That UI is for local operations and demos. The root Docker/Space deployment is API-first for the OpenEnv validator.

## Docker

Build and run:

```bash
docker build -t support-triage-env .
docker run --rm -p 8000:8000 support-triage-env
```

The container starts `uvicorn server.app:app` on port `8000`.

## Hugging Face Spaces

This repository is configured for a Docker Space that serves the OpenEnv API:

- `README.md` uses `sdk: docker`
- `app_port: 8000`
- root `Dockerfile` launches the FastAPI/OpenEnv server
- `openenv.yaml` points to `server.app:app`

Recommended Space secrets / variables:

```bash
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_provider_key
```

If you want to use OpenAI instead, switch `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` accordingly.

## Project Structure

```text
.
├── README.md
├── Dockerfile
├── openenv.yaml
├── inference.py
├── baseline.py
├── llm_client.py
├── models.py
├── tasks.py
├── graders.py
├── client.py
├── groq_reply.py
├── streamlit_app.py
├── server/
│   ├── app.py
│   ├── Dockerfile
│   └── support_triage_environment.py
└── tests/
    └── test_environment.py
```
