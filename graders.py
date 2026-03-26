"""Deterministic graders for the support triage environment."""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field

try:
    from .models import SupportTriageState, TicketState
    from .tasks import CUSTOM_TASK_ID, TicketOutcome, get_task
except ImportError:
    from models import SupportTriageState, TicketState
    from tasks import CUSTOM_TASK_ID, TicketOutcome, get_task

DIGIT_RUN_RE = re.compile(r"\d(?:[\s-]*\d){11,}")


class GraderResult(BaseModel):
    """Structured episode grade returned by local graders and the /grader endpoint."""

    task_id: str = Field(..., description="Task identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall task score")
    passed: bool = Field(..., description="Whether the agent reached a strong score")
    breakdown: dict[str, float] = Field(
        default_factory=dict, description="Normalized component scores"
    )
    details: dict[str, object] = Field(
        default_factory=dict, description="Extra deterministic grading metadata"
    )


def _normalize_keywords(text: str | None) -> str:
    return (text or "").strip().lower()


def _keyword_coverage(text: str | None, keywords: Iterable[str]) -> float:
    normalized = _normalize_keywords(text)
    words = tuple(keyword.lower() for keyword in keywords)
    if not words:
        return 1.0
    matched = sum(1 for keyword in words if keyword in normalized)
    return matched / len(words)


def _reply_penalty(ticket: TicketState, outcome: TicketOutcome) -> float:
    reply_text = _normalize_keywords(ticket.public_reply)
    penalty = 0.0
    if outcome.forbidden_reply_keywords and reply_text:
        forbidden_hits = sum(
            1 for keyword in outcome.forbidden_reply_keywords if keyword in reply_text
        )
        if forbidden_hits:
            penalty += min(0.5, 0.25 * forbidden_hits)
    if outcome.forbid_digit_echo and reply_text and DIGIT_RUN_RE.search(reply_text):
        penalty += 0.75
    return penalty


def _score_ticket(ticket: TicketState, outcome: TicketOutcome) -> tuple[float, float, dict[str, float]]:
    achieved = 0.0
    total = 0.0
    detail: dict[str, float] = {}

    def score_component(name: str, current: float, weight: float) -> None:
        nonlocal achieved, total
        detail[name] = round(current, 4)
        achieved += current * weight
        total += weight

    if outcome.must_open:
        score_component("opened", 1.0 if ticket.opened else 0.0, 0.75)

    score_component("priority", 1.0 if ticket.priority == outcome.priority else 0.0, 1.25)
    score_component("queue", 1.0 if ticket.queue == outcome.queue else 0.0, 1.25)
    score_component("status", 1.0 if ticket.status == outcome.status else 0.0, 1.0)

    if outcome.required_tags:
        required = set(outcome.required_tags)
        applied = set(ticket.tags)
        tag_score = len(required & applied) / len(required)
        score_component("tags", tag_score, 1.0)

    if outcome.required_reply_keywords:
        reply_score = _keyword_coverage(ticket.public_reply, outcome.required_reply_keywords)
        reply_score = max(0.0, reply_score - _reply_penalty(ticket, outcome))
        score_component("reply", reply_score, 1.25)

    if outcome.required_note_keywords:
        note_score = _keyword_coverage(ticket.internal_note, outcome.required_note_keywords)
        score_component("note", note_score, 0.75)

    if outcome.redaction_required:
        score_component("redaction", 1.0 if ticket.redaction_applied else 0.0, 1.0)

    return achieved, total, detail


def grade_support_episode(state: SupportTriageState) -> GraderResult:
    """Score an episode deterministically from the serialized state."""

    if state.task_id == CUSTOM_TASK_ID:
        ticket = state.tickets[0] if state.tickets else None
        if ticket is None:
            return GraderResult(
                task_id=state.task_id,
                score=0.0,
                passed=False,
                breakdown={"ticket:missing": 0.0},
                details={},
            )

        opened_score = 1.0 if ticket.opened else 0.0
        reply_score = 1.0 if ticket.public_reply_sent and (ticket.public_reply or "").strip() else 0.0
        safe_reply_score = 1.0
        if ticket.public_reply_sent and DIGIT_RUN_RE.search(_normalize_keywords(ticket.public_reply)):
            safe_reply_score = 0.0
        redaction_score = 1.0 if not ticket.contains_sensitive_data or ticket.redaction_applied else 0.0

        achieved = (opened_score * 0.8) + (reply_score * 1.2) + (safe_reply_score * 0.6) + (redaction_score * 0.8)
        total = 3.4
        final_score = round(achieved / total, 4)
        return GraderResult(
            task_id=state.task_id,
            score=final_score,
            passed=final_score >= 0.75,
            breakdown={
                f"ticket:{ticket.ticket_id}": round((opened_score + reply_score + safe_reply_score + redaction_score) / 4, 4),
                "workflow:opened": round(opened_score, 4),
                "workflow:reply_sent": round(reply_score, 4),
                "workflow:safe_reply": round(safe_reply_score, 4),
                "workflow:redaction_or_safe_content": round(redaction_score, 4),
            },
            details={
                "ticket_id": ticket.ticket_id,
                "opened": ticket.opened,
                "public_reply_sent": ticket.public_reply_sent,
                "redaction_applied": ticket.redaction_applied,
                "contains_sensitive_data": ticket.contains_sensitive_data,
            },
        )

    task = get_task(state.task_id)
    ticket_map = {ticket.ticket_id: ticket for ticket in state.tickets}
    achieved = 0.0
    total = 0.0
    breakdown: dict[str, float] = {}
    details: dict[str, object] = {"tickets": {}}

    for ticket_id, outcome in task.expected.items():
        ticket = ticket_map[ticket_id]
        ticket_achieved, ticket_total, ticket_detail = _score_ticket(ticket, outcome)
        achieved += ticket_achieved
        total += ticket_total
        normalized_ticket = ticket_achieved / ticket_total if ticket_total else 1.0
        breakdown[f"ticket:{ticket_id}"] = round(normalized_ticket, 4)
        details["tickets"][ticket_id] = ticket_detail

    if task.first_decision_ticket_id:
        first_decision = state.decision_ticket_order[0] if state.decision_ticket_order else None
        order_score = 1.0 if first_decision == task.first_decision_ticket_id else 0.0
        achieved += order_score * 1.0
        total += 1.0
        breakdown["workflow:first_decision"] = round(order_score, 4)
        details["first_decision_ticket_id"] = first_decision

    if task.redact_before_reply_ticket_ids:
        ordering_scores: list[float] = []
        for ticket_id in task.redact_before_reply_ticket_ids:
            redact_index = None
            reply_index = None
            for idx, record in enumerate(state.action_history):
                if record.ticket_id != ticket_id:
                    continue
                if record.action_type == "redact_sensitive_data" and redact_index is None:
                    redact_index = idx
                if record.action_type == "reply_to_ticket" and reply_index is None:
                    reply_index = idx
            score = 1.0 if redact_index is not None and (reply_index is None or redact_index < reply_index) else 0.0
            ordering_scores.append(score)
            details[f"redact_before_reply:{ticket_id}"] = {
                "redact_index": redact_index,
                "reply_index": reply_index,
            }
        workflow_score = sum(ordering_scores) / len(ordering_scores)
        achieved += workflow_score * 1.0
        total += 1.0
        breakdown["workflow:redact_before_reply"] = round(workflow_score, 4)

    final_score = round(achieved / total if total else 0.0, 4)
    return GraderResult(
        task_id=state.task_id,
        score=final_score,
        passed=final_score >= 0.85,
        breakdown=breakdown,
        details=details,
    )
