"""Deterministic task bank for the support triage environment."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TicketSeed:
    """Immutable starting state for a ticket."""

    ticket_id: str
    customer_tier: str
    channel: str
    subject: str
    body: str
    hours_open: int
    contains_sensitive_data: bool = False


@dataclass(frozen=True)
class TicketOutcome:
    """Expected final state for a ticket."""

    priority: str
    queue: str
    status: str
    required_tags: tuple[str, ...] = ()
    required_reply_keywords: tuple[str, ...] = ()
    forbidden_reply_keywords: tuple[str, ...] = ()
    required_note_keywords: tuple[str, ...] = ()
    redaction_required: bool = False
    forbid_digit_echo: bool = False
    must_open: bool = True


@dataclass(frozen=True)
class TaskDefinition:
    """Full scenario definition including hidden grading rubric."""

    task_id: str
    title: str
    difficulty: str
    objective: str
    success_criteria: tuple[str, ...]
    max_steps: int
    tickets: tuple[TicketSeed, ...]
    expected: dict[str, TicketOutcome]
    first_decision_ticket_id: str | None = None
    redact_before_reply_ticket_ids: tuple[str, ...] = ()


TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        task_id="billing_refund_easy",
        title="Easy Billing Refund Escalation",
        difficulty="easy",
        objective=(
            "Triage a single billing complaint, classify it correctly, and send a safe "
            "acknowledgement before escalating the ticket."
        ),
        success_criteria=(
            "Inspect the customer message before taking action.",
            "Route the issue to billing with the correct urgency and refund tags.",
            "Send a concise acknowledgement without overpromising the outcome.",
        ),
        max_steps=8,
        tickets=(
            TicketSeed(
                ticket_id="BILL-1001",
                customer_tier="pro",
                channel="email",
                subject="Charged after cancellation",
                body=(
                    "I canceled yesterday, but I still see a duplicate $49 charge. "
                    "Can someone look into a refund for me?"
                ),
                hours_open=5,
            ),
        ),
        expected={
            "BILL-1001": TicketOutcome(
                priority="high",
                queue="billing",
                status="escalated",
                required_tags=("billing", "refund"),
                required_reply_keywords=("sorry", "refund", "billing"),
                forbidden_reply_keywords=("guarantee",),
            )
        },
    ),
    TaskDefinition(
        task_id="security_and_invoice_medium",
        title="Medium Security and Invoice Split",
        difficulty="medium",
        objective=(
            "Handle an urgent account-compromise report before resolving a routine "
            "billing question in the same queue."
        ),
        success_criteria=(
            "Handle the urgent security incident first.",
            "Escalate the security issue and reassure the customer with a safe reply.",
            "Resolve the invoice request cleanly without escalating it unnecessarily.",
        ),
        max_steps=14,
        tickets=(
            TicketSeed(
                ticket_id="SEC-2201",
                customer_tier="enterprise",
                channel="email",
                subject="Unknown login from Moscow",
                body=(
                    "We saw a login from Moscow that nobody on our team recognizes. "
                    "Please lock the account and escalate this immediately."
                ),
                hours_open=1,
            ),
            TicketSeed(
                ticket_id="BILL-2202",
                customer_tier="starter",
                channel="chat",
                subject="Need invoice copy for accounting",
                body=(
                    "Could you point me to where I can download the latest invoice for "
                    "our records?"
                ),
                hours_open=18,
            ),
        ),
        expected={
            "SEC-2201": TicketOutcome(
                priority="urgent",
                queue="security",
                status="escalated",
                required_tags=("security", "account_compromise"),
                required_reply_keywords=("security", "escalated", "lock"),
            ),
            "BILL-2202": TicketOutcome(
                priority="normal",
                queue="billing",
                status="resolved",
                required_tags=("billing", "invoice"),
                required_reply_keywords=("invoice", "portal"),
            ),
        },
        first_decision_ticket_id="SEC-2201",
    ),
    TaskDefinition(
        task_id="vip_incident_hard",
        title="Hard VIP Incident Queue",
        difficulty="hard",
        objective=(
            "Work a mixed VIP incident queue with a leaked API key, raw card data in a "
            "refund request, and a low-priority product request."
        ),
        success_criteria=(
            "Prioritize the live security exposure before lower-risk work.",
            "Redact sensitive customer data before replying on the billing-risk ticket.",
            "Leave crisp customer communication and internal notes that match the right workflow.",
        ),
        max_steps=20,
        tickets=(
            TicketSeed(
                ticket_id="SEC-3301",
                customer_tier="enterprise",
                channel="email",
                subject="API key leaked in screenshot",
                body=(
                    "I posted a screenshot publicly and it accidentally shows our live API "
                    "key. Please help us lock this down now."
                ),
                hours_open=1,
                contains_sensitive_data=True,
            ),
            TicketSeed(
                ticket_id="BILL-3302",
                customer_tier="vip",
                channel="email",
                subject="Refund request with full card number",
                body=(
                    "I was charged twice and my card was 4242 4242 4242 4242. "
                    "Please process a refund as soon as possible."
                ),
                hours_open=2,
                contains_sensitive_data=True,
            ),
            TicketSeed(
                ticket_id="PROD-3303",
                customer_tier="starter",
                channel="chat",
                subject="Feature request for export dark mode",
                body=(
                    "It would be great if the PDF export respected our dark mode theme. "
                    "Can you pass this to the product team?"
                ),
                hours_open=20,
            ),
        ),
        expected={
            "SEC-3301": TicketOutcome(
                priority="urgent",
                queue="security",
                status="escalated",
                required_tags=("security", "credential_exposure"),
                required_reply_keywords=("rotate", "security", "escalated"),
                required_note_keywords=("revoke", "audit"),
            ),
            "BILL-3302": TicketOutcome(
                priority="high",
                queue="billing_risk",
                status="escalated",
                required_tags=("billing", "refund", "pii"),
                required_reply_keywords=("secure", "billing", "escalated"),
                redaction_required=True,
                forbid_digit_echo=True,
            ),
            "PROD-3303": TicketOutcome(
                priority="low",
                queue="product",
                status="triaged",
                required_tags=("feature_request",),
                required_reply_keywords=("roadmap", "product"),
            ),
        },
        first_decision_ticket_id="SEC-3301",
        redact_before_reply_ticket_ids=("BILL-3302",),
    ),
)

TASKS_BY_ID = {task.task_id: task for task in TASKS}
DEFAULT_TASK_ID = TASKS[0].task_id
CUSTOM_TASK_ID = "custom_message_sandbox"
DEFAULT_CUSTOM_TICKET_ID = "CUSTOM-1001"
CUSTOM_TASK_CARD = {
    "task_id": CUSTOM_TASK_ID,
    "title": "Custom Message Sandbox",
    "difficulty": "easy",
    "objective": (
        "Create a one-off custom support ticket so you can test opening, redacting, "
        "and replying to your own customer message."
    ),
    "success_criteria": [
        "Reset with custom_subject and custom_body.",
        "Open the ticket before editing or replying.",
        "Send a safe manual or OpenAI-generated customer reply.",
    ],
    "max_steps": 10,
    "ticket_count": 1,
}

SENSITIVE_DIGIT_RE = re.compile(r"\d(?:[\s-]*\d){11,}")
SECRET_HINT_RE = re.compile(r"\b(api key|secret|token|password|credential)\b", re.IGNORECASE)


def get_task(task_id: str) -> TaskDefinition:
    """Return a task definition or raise a helpful error."""
    if task_id == CUSTOM_TASK_ID:
        return TaskDefinition(
            task_id=CUSTOM_TASK_ID,
            title=str(CUSTOM_TASK_CARD["title"]),
            difficulty=str(CUSTOM_TASK_CARD["difficulty"]),
            objective=str(CUSTOM_TASK_CARD["objective"]),
            success_criteria=tuple(CUSTOM_TASK_CARD["success_criteria"]),
            max_steps=int(CUSTOM_TASK_CARD["max_steps"]),
            tickets=(),
            expected={},
        )
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:
        valid = ", ".join(sorted([*TASKS_BY_ID, CUSTOM_TASK_ID]))
        raise KeyError(f"Unknown task_id '{task_id}'. Valid task_ids: {valid}") from exc


def public_task_cards() -> list[dict[str, object]]:
    """Return public task metadata suitable for API responses and docs."""
    cards = [
        {
            "task_id": task.task_id,
            "title": task.title,
            "difficulty": task.difficulty,
            "objective": task.objective,
            "success_criteria": list(task.success_criteria),
            "max_steps": task.max_steps,
            "ticket_count": len(task.tickets),
        }
        for task in TASKS
    ]
    cards.append(dict(CUSTOM_TASK_CARD))
    return cards


def infer_sensitive_content(text: str) -> bool:
    """Best-effort heuristic for custom sandbox tickets."""

    return bool(SENSITIVE_DIGIT_RE.search(text) or SECRET_HINT_RE.search(text))
