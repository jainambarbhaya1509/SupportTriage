"""Environment implementation for support queue triage."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..graders import grade_support_episode
    from ..openai_reply import generate_llm_reply
    from ..models import (
        ActionRecord,
        ActiveTicket,
        InboxTicket,
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TicketState,
    )
    from ..tasks import (
        CUSTOM_TASK_CARD,
        CUSTOM_TASK_ID,
        DEFAULT_CUSTOM_TICKET_ID,
        DEFAULT_TASK_ID,
        TicketSeed,
        get_task,
        infer_sensitive_content,
    )
except ImportError:
    from graders import grade_support_episode
    from openai_reply import generate_llm_reply
    from models import (
        ActionRecord,
        ActiveTicket,
        InboxTicket,
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TicketState,
    )
    from tasks import (
        CUSTOM_TASK_CARD,
        CUSTOM_TASK_ID,
        DEFAULT_CUSTOM_TICKET_ID,
        DEFAULT_TASK_ID,
        TicketSeed,
        get_task,
        infer_sensitive_content,
    )


class SupportTriageEnvironment(
    Environment[SupportTriageAction, SupportTriageObservation, SupportTriageState]
):
    """Deterministic support-operations environment with dense rewards."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = SupportTriageState()
        self._current_task_id = DEFAULT_TASK_ID
        self._last_progress_score = 0.0
        self._last_action_signature: str | None = None
        self._last_completion_summary = ""
        self.reset()

    def _build_ticket_state(self, seed: TicketSeed) -> TicketState:
        return TicketState(
            ticket_id=seed.ticket_id,
            customer_tier=seed.customer_tier,  # type: ignore[arg-type]
            channel=seed.channel,  # type: ignore[arg-type]
            subject=seed.subject,
            body=seed.body,
            hours_open=seed.hours_open,
            contains_sensitive_data=seed.contains_sensitive_data,
        )

    def _ticket_to_inbox(self, ticket: TicketState) -> InboxTicket:
        return InboxTicket.model_validate(
            ticket.model_dump(
                exclude={"body", "public_reply", "internal_note", "contains_sensitive_data"}
            )
        )

    def _ticket_to_active(self, ticket: TicketState) -> ActiveTicket:
        return ActiveTicket.model_validate(
            ticket.model_dump(exclude={"contains_sensitive_data"})
        )

    def _find_ticket(self, ticket_id: str | None) -> TicketState | None:
        if not ticket_id:
            return None
        for ticket in self._state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _progress_score(self) -> float:
        return grade_support_episode(self._state).score

    def _remaining_steps(self) -> int:
        return max(0, self._state.max_steps - self._state.step_count)

    def _observation(self, feedback: str, reward: float, done: bool) -> SupportTriageObservation:
        active_ticket = self._find_ticket(self._state.active_ticket_id)
        return SupportTriageObservation(
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            difficulty=self._state.difficulty,
            objective=self._state.objective,
            success_criteria=list(self._state.success_criteria),
            inbox=[self._ticket_to_inbox(ticket) for ticket in self._state.tickets],
            active_ticket=self._ticket_to_active(active_ticket) if active_ticket else None,
            action_feedback=feedback,
            remaining_steps=self._remaining_steps(),
            completion_score=round(self._state.score, 4),
            invalid_action_count=self._state.invalid_action_count,
            cumulative_reward=round(self._state.cumulative_reward, 4),
            reward=round(reward, 4),
            done=done,
            metadata={
                "task_id": self._state.task_id,
                "step_count": self._state.step_count,
                "completion_summary": self._last_completion_summary,
            },
        )

    def _record_action(
        self,
        action: SupportTriageAction,
        summary: str,
        reward: float,
        *,
        invalid: bool = False,
    ) -> None:
        self._state.action_history.append(
            ActionRecord(
                step_index=self._state.step_count,
                action_type=action.action_type,
                ticket_id=action.ticket_id,
                summary=summary,
                reward=round(reward, 4),
                score_after_action=round(self._state.score, 4),
                invalid=invalid,
            )
        )

    def _apply_open_ticket(self, action: SupportTriageAction) -> tuple[bool, str, float]:
        ticket = self._find_ticket(action.ticket_id)
        if ticket is None:
            return True, f"Ticket {action.ticket_id} does not exist.", -0.20

        loop_penalty = 0.0
        if self._state.active_ticket_id == ticket.ticket_id and ticket.opened:
            self._state.loop_penalty_count += 1
            loop_penalty = 0.05

        ticket.opened = True
        self._state.active_ticket_id = ticket.ticket_id
        return False, f"Opened {ticket.ticket_id}: {ticket.subject}", -loop_penalty

    def _require_active_ticket(self, ticket_id: str | None) -> tuple[TicketState | None, str | None]:
        ticket = self._find_ticket(ticket_id)
        if ticket is None:
            return None, f"Ticket {ticket_id} does not exist."
        if not ticket.opened:
            return None, f"Open {ticket.ticket_id} before editing or replying."
        if self._state.active_ticket_id != ticket.ticket_id:
            return None, f"Ticket {ticket.ticket_id} is not currently active. Open it first."
        return ticket, None

    def _apply_update_ticket(self, action: SupportTriageAction) -> tuple[bool, str, float]:
        ticket, error = self._require_active_ticket(action.ticket_id)
        if ticket is None:
            return True, error or "Invalid update.", -0.20

        changes: list[str] = []
        if action.priority and action.priority != ticket.priority:
            ticket.priority = action.priority
            changes.append(f"priority={action.priority}")
        if action.queue and action.queue != ticket.queue:
            ticket.queue = action.queue
            changes.append(f"queue={action.queue}")
        if action.status and action.status != ticket.status:
            ticket.status = action.status
            changes.append(f"status={action.status}")
        if action.tags:
            normalized_tags = sorted(dict.fromkeys(action.tags))
            if normalized_tags != sorted(ticket.tags):
                ticket.tags = normalized_tags
                changes.append(f"tags={','.join(normalized_tags)}")

        if not changes:
            self._state.loop_penalty_count += 1
            return False, f"No effective changes were made to {ticket.ticket_id}.", -0.08

        if action.ticket_id not in self._state.decision_ticket_order:
            self._state.decision_ticket_order.append(action.ticket_id)
        return False, f"Updated {ticket.ticket_id}: {', '.join(changes)}", 0.0

    def _apply_reply(self, action: SupportTriageAction) -> tuple[bool, str, float]:
        ticket, error = self._require_active_ticket(action.ticket_id)
        if ticket is None:
            return True, error or "Invalid reply.", -0.25

        penalty = 0.0
        if ticket.contains_sensitive_data:
            if self._state.task_id == CUSTOM_TASK_ID:
                if not ticket.redaction_applied:
                    penalty += 0.20
            else:
                task = get_task(self._state.task_id)
                outcome = task.expected[ticket.ticket_id]
                if outcome.redaction_required and not ticket.redaction_applied:
                    penalty += 0.20

        provider_used: str | None = None
        message = (action.message or "").strip()
        if not message:
            try:
                observation = self._observation("", reward=0.0, done=False)
                message, provider_used = generate_llm_reply(
                    observation=observation,
                    ticket_id=ticket.ticket_id,
                )
                message = message.strip()
            except ValueError as exc:
                return True, str(exc), -0.20
            except Exception as exc:
                return True, f"OpenAI reply generation failed: {exc}", -0.25

            if not message:
                return True, "OpenAI returned an empty reply.", -0.20

        if self._state.task_id == "vip_incident_hard" and any(char.isdigit() for char in message):
            penalty += 0.15

        ticket.public_reply = message
        ticket.public_reply_sent = True
        if action.ticket_id not in self._state.decision_ticket_order:
            self._state.decision_ticket_order.append(action.ticket_id)
        if provider_used:
            return False, f"Sent {provider_used}-generated reply on {ticket.ticket_id}.", -penalty
        return False, f"Sent reply on {ticket.ticket_id}.", -penalty

    def _apply_note(self, action: SupportTriageAction) -> tuple[bool, str, float]:
        ticket, error = self._require_active_ticket(action.ticket_id)
        if ticket is None:
            return True, error or "Invalid note.", -0.20

        note = (action.note or "").strip()
        if not note:
            return True, "Internal note cannot be empty.", -0.20

        ticket.internal_note = note
        ticket.internal_note_present = True
        if action.ticket_id not in self._state.decision_ticket_order:
            self._state.decision_ticket_order.append(action.ticket_id)
        return False, f"Added internal note on {ticket.ticket_id}.", 0.0

    def _apply_redaction(self, action: SupportTriageAction) -> tuple[bool, str, float]:
        ticket, error = self._require_active_ticket(action.ticket_id)
        if ticket is None:
            return True, error or "Invalid redaction.", -0.25

        if ticket.redaction_applied:
            self._state.loop_penalty_count += 1
            return False, f"{ticket.ticket_id} has already been redacted.", -0.08

        if not ticket.contains_sensitive_data:
            return True, f"{ticket.ticket_id} does not contain sensitive data to redact.", -0.15

        ticket.redaction_applied = True
        ticket.contains_sensitive_data = False
        ticket.body = "[REDACTED SENSITIVE DATA] " + ticket.body
        if action.ticket_id not in self._state.decision_ticket_order:
            self._state.decision_ticket_order.append(action.ticket_id)
        return False, f"Applied redaction to {ticket.ticket_id}.", 0.0

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        custom_ticket_id: str | None = None,
        custom_subject: str | None = None,
        custom_body: str | None = None,
        custom_customer_tier: str | None = None,
        custom_channel: str | None = None,
        custom_hours_open: int | None = None,
        custom_contains_sensitive_data: bool | None = None,
        **_: Any,
    ) -> SupportTriageObservation:
        requested_task_id = task_id or DEFAULT_TASK_ID
        if requested_task_id == CUSTOM_TASK_ID or custom_body:
            body = (custom_body or "").strip()
            if not body:
                raise ValueError(
                    "custom_body is required when resetting the custom_message_sandbox task."
                )

            subject = (custom_subject or "Custom support request").strip()
            ticket_id_value = (custom_ticket_id or DEFAULT_CUSTOM_TICKET_ID).strip()
            customer_tier = (custom_customer_tier or "pro").strip()
            channel = (custom_channel or "email").strip()
            hours_open = custom_hours_open if custom_hours_open is not None else 0
            contains_sensitive = (
                custom_contains_sensitive_data
                if custom_contains_sensitive_data is not None
                else infer_sensitive_content(body)
            )

            self._current_task_id = CUSTOM_TASK_ID
            self._state = SupportTriageState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                task_id=CUSTOM_TASK_ID,
                task_title=str(CUSTOM_TASK_CARD["title"]),
                difficulty="easy",
                objective=str(CUSTOM_TASK_CARD["objective"]),
                success_criteria=list(CUSTOM_TASK_CARD["success_criteria"]),
                max_steps=int(CUSTOM_TASK_CARD["max_steps"]),
                tickets=[
                    self._build_ticket_state(
                        TicketSeed(
                            ticket_id=ticket_id_value,
                            customer_tier=customer_tier,
                            channel=channel,
                            subject=subject,
                            body=body,
                            hours_open=hours_open,
                            contains_sensitive_data=contains_sensitive,
                        )
                    )
                ],
                latest_feedback=(
                    f"Loaded custom ticket '{ticket_id_value}'. Start by opening it to inspect the full message."
                ),
            )
        else:
            task = get_task(requested_task_id)
            self._current_task_id = task.task_id
            self._state = SupportTriageState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                task_id=task.task_id,
                task_title=task.title,
                difficulty=task.difficulty,  # type: ignore[arg-type]
                objective=task.objective,
                success_criteria=list(task.success_criteria),
                max_steps=task.max_steps,
                tickets=[self._build_ticket_state(ticket) for ticket in task.tickets],
                latest_feedback=(
                    f"Loaded task '{task.title}'. Start by opening a ticket to inspect it."
                ),
            )
        self._last_progress_score = self._progress_score()
        self._state.score = self._last_progress_score
        self._last_action_signature = None
        self._last_completion_summary = ""
        return self._observation(self._state.latest_feedback, reward=0.0, done=False)

    def step(
        self,
        action: SupportTriageAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> SupportTriageObservation:
        del timeout_s
        self._state.step_count += 1
        invalid = False
        done = False
        base_penalty = 0.02

        action_signature = json.dumps(action.model_dump(mode="json"), sort_keys=True)
        repeated_penalty = 0.0
        if action_signature == self._last_action_signature:
            self._state.loop_penalty_count += 1
            repeated_penalty = 0.05
        self._last_action_signature = action_signature

        if action.action_type == "open_ticket":
            invalid, feedback, direct_adjustment = self._apply_open_ticket(action)
        elif action.action_type == "update_ticket":
            invalid, feedback, direct_adjustment = self._apply_update_ticket(action)
        elif action.action_type == "reply_to_ticket":
            invalid, feedback, direct_adjustment = self._apply_reply(action)
        elif action.action_type == "add_internal_note":
            invalid, feedback, direct_adjustment = self._apply_note(action)
        elif action.action_type == "redact_sensitive_data":
            invalid, feedback, direct_adjustment = self._apply_redaction(action)
        elif action.action_type == "complete_episode":
            done = True
            self._state.submitted = True
            self._last_completion_summary = action.summary or "Agent completed the episode."
            feedback = self._last_completion_summary
            direct_adjustment = 0.0
        else:
            invalid = True
            feedback = f"Unsupported action type: {action.action_type}"
            direct_adjustment = -0.25

        if invalid:
            self._state.invalid_action_count += 1

        progress_score = self._progress_score()
        delta_progress = progress_score - self._last_progress_score
        reward = round(delta_progress + direct_adjustment - base_penalty - repeated_penalty, 4)
        self._last_progress_score = progress_score
        self._state.score = progress_score
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)
        self._state.latest_feedback = feedback

        if self._state.step_count >= self._state.max_steps:
            done = True
            if not self._state.submitted:
                feedback = f"{feedback} Step limit reached."
                self._state.latest_feedback = feedback

        self._record_action(action, feedback, reward, invalid=invalid)
        return self._observation(feedback, reward=reward, done=done)

    @property
    def state(self) -> SupportTriageState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="support_triage_env",
            description=(
                "A deterministic support-operations inbox where agents triage "
                "billing, security, and product tickets with dense rewards."
            ),
            version="0.1.0",
            author="Codex",
        )
