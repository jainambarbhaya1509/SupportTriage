"""Typed models for the support triage environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator

TicketPriority = Literal["low", "normal", "high", "urgent"]
TicketQueue = Literal["general", "billing", "billing_risk", "security", "product"]
TicketStatus = Literal[
    "new",
    "triaged",
    "waiting_on_customer",
    "resolved",
    "escalated",
    "closed",
]
CustomerTier = Literal["starter", "pro", "enterprise", "vip"]
SupportChannel = Literal["email", "chat"]
Difficulty = Literal["easy", "medium", "hard"]
ActionType = Literal[
    "open_ticket",
    "update_ticket",
    "reply_to_ticket",
    "add_internal_note",
    "redact_sensitive_data",
    "complete_episode",
]


class InboxTicket(BaseModel):
    """Compact ticket summary visible in the inbox."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_tier: CustomerTier = Field(..., description="Customer account tier")
    channel: SupportChannel = Field(..., description="Source channel for the ticket")
    subject: str = Field(..., description="Ticket subject line")
    hours_open: int = Field(..., ge=0, description="Age of the ticket in hours")
    priority: TicketPriority = Field(
        default="normal", description="Current priority chosen by the agent"
    )
    queue: TicketQueue = Field(
        default="general", description="Current queue assignment"
    )
    status: TicketStatus = Field(default="new", description="Current workflow state")
    tags: list[str] = Field(default_factory=list, description="Applied ticket tags")
    opened: bool = Field(
        default=False, description="Whether the agent has inspected the ticket body"
    )
    redaction_applied: bool = Field(
        default=False, description="Whether a sensitive-data redaction was applied"
    )
    public_reply_sent: bool = Field(
        default=False, description="Whether a customer-visible reply has been sent"
    )
    internal_note_present: bool = Field(
        default=False, description="Whether an internal note has been added"
    )


class ActiveTicket(InboxTicket):
    """Expanded ticket detail shown when a ticket is opened."""

    body: str = Field(..., description="Full customer message")
    public_reply: str | None = Field(
        default=None, description="Most recent public reply drafted by the agent"
    )
    internal_note: str | None = Field(
        default=None, description="Most recent internal note drafted by the agent"
    )


class TicketState(ActiveTicket):
    """Internal ticket state used by the environment and graders."""

    contains_sensitive_data: bool = Field(
        default=False, description="Whether the ticket currently contains raw PII/secrets"
    )


class ActionRecord(BaseModel):
    """Audit trail entry for a single agent action."""

    step_index: int = Field(..., ge=0, description="1-based step index")
    action_type: ActionType = Field(..., description="Action executed by the agent")
    ticket_id: str | None = Field(
        default=None, description="Ticket targeted by the action when applicable"
    )
    summary: str = Field(..., description="Human-readable summary of the action")
    reward: float = Field(default=0.0, description="Reward issued after the action")
    score_after_action: float = Field(
        default=0.0, description="Grader score after the action"
    )
    invalid: bool = Field(
        default=False, description="Whether the environment treated the action as invalid"
    )


class SupportTriageAction(Action):
    """Typed action schema for the environment."""

    action_type: ActionType = Field(..., description="Kind of support workflow action")
    ticket_id: str | None = Field(
        default=None, description="Target ticket id for ticket-scoped actions"
    )
    priority: TicketPriority | None = Field(
        default=None, description="New priority to assign during update_ticket"
    )
    queue: TicketQueue | None = Field(
        default=None, description="New queue to assign during update_ticket"
    )
    status: TicketStatus | None = Field(
        default=None, description="New workflow status during update_ticket"
    )
    tags: list[str] = Field(
        default_factory=list, description="Ticket tags to set during update_ticket"
    )
    message: str | None = Field(
        default=None,
        description=(
            "Customer-visible reply for reply_to_ticket. If omitted and the server has "
            "GROQ_API_KEY configured, the backend may generate the reply automatically."
        ),
    )
    note: str | None = Field(
        default=None, description="Internal note text for add_internal_note"
    )
    reason: str | None = Field(
        default=None, description="Reason used when redacting sensitive data"
    )
    summary: str | None = Field(
        default=None, description="Short wrap-up for complete_episode"
    )

    @model_validator(mode="after")
    def validate_action(self) -> "SupportTriageAction":
        if self.action_type == "complete_episode":
            return self

        if not self.ticket_id:
            raise ValueError(f"{self.action_type} requires ticket_id")

        if self.action_type == "update_ticket":
            if not any(
                value is not None
                for value in (self.priority, self.queue, self.status)
            ) and not self.tags:
                raise ValueError(
                    "update_ticket requires at least one of priority, queue, status, or tags"
                )
        elif self.action_type == "add_internal_note" and not self.note:
            raise ValueError("add_internal_note requires note")
        elif self.action_type == "redact_sensitive_data" and not self.reason:
            raise ValueError("redact_sensitive_data requires reason")
        return self


class SupportTriageObservation(Observation):
    """Observation returned after reset and each environment step."""

    task_id: str = Field(..., description="Scenario identifier")
    task_title: str = Field(..., description="Human-readable task title")
    difficulty: Difficulty = Field(..., description="Task difficulty level")
    objective: str = Field(..., description="High-level objective for the current task")
    success_criteria: list[str] = Field(
        default_factory=list, description="Public success criteria for the task"
    )
    inbox: list[InboxTicket] = Field(
        default_factory=list, description="Visible inbox summary for all tickets"
    )
    active_ticket: ActiveTicket | None = Field(
        default=None, description="Expanded ticket currently opened by the agent"
    )
    action_feedback: str = Field(
        default="", description="Immediate feedback on the previous action"
    )
    remaining_steps: int = Field(
        default=0, ge=0, description="How many steps remain in the episode"
    )
    completion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current grader score based on the state so far",
    )
    invalid_action_count: int = Field(
        default=0, ge=0, description="Number of invalid actions issued so far"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Running sum of per-step rewards"
    )


class SupportTriageState(State):
    """Serializable internal state exposed by state()."""

    task_id: str = Field(default="", description="Scenario identifier")
    task_title: str = Field(default="", description="Task title")
    difficulty: Difficulty = Field(default="easy", description="Task difficulty")
    objective: str = Field(default="", description="Task objective")
    success_criteria: list[str] = Field(
        default_factory=list, description="Public success criteria"
    )
    active_ticket_id: str | None = Field(
        default=None, description="Currently opened ticket id"
    )
    max_steps: int = Field(default=0, ge=0, description="Maximum steps in episode")
    cumulative_reward: float = Field(
        default=0.0, description="Running reward total"
    )
    invalid_action_count: int = Field(
        default=0, ge=0, description="Invalid action counter"
    )
    loop_penalty_count: int = Field(
        default=0, ge=0, description="Repeated or no-op action counter"
    )
    submitted: bool = Field(
        default=False, description="Whether the agent explicitly completed the episode"
    )
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current grader score"
    )
    latest_feedback: str = Field(
        default="", description="Feedback attached to the latest step"
    )
    decision_ticket_order: list[str] = Field(
        default_factory=list,
        description="Ticket ids touched by decision actions in chronological order",
    )
    tickets: list[TicketState] = Field(
        default_factory=list, description="Full mutable ticket state"
    )
    action_history: list[ActionRecord] = Field(
        default_factory=list, description="Chronological action history"
    )

    def active_ticket(self) -> TicketState | None:
        """Return the active ticket, if any."""
        if not self.active_ticket_id:
            return None
        for ticket in self.tickets:
            if ticket.ticket_id == self.active_ticket_id:
                return ticket
        return None


def model_to_jsonable(model: BaseModel) -> dict[str, Any]:
    """Serialize a pydantic model for prompts and API helpers."""
    return model.model_dump(mode="json")
