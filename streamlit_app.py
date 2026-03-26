"""Streamlit operator console for the support triage environment."""

from __future__ import annotations

import html
import os
import time
from typing import Any, get_args

import streamlit as st
import streamlit.components.v1 as components

from baseline import DEFAULT_MODEL, run_baseline_sync
from graders import grade_support_episode
from mail_bridge import (
    MailProviderConfig,
    dismiss_approval_item,
    load_mail_config_from_env,
    pending_approval_items,
    regenerate_approval_draft,
    send_approved_reply,
    sync_inbox_for_approval,
)
from models import (
    SupportTriageAction,
    SupportTriageObservation,
    TicketPriority,
    TicketQueue,
    TicketStatus,
)
from server.support_triage_environment import SupportTriageEnvironment
from tasks import CUSTOM_TASK_ID, infer_sensitive_content, public_task_cards

ACTION_HELP = {
    "open_ticket": "Open a ticket so the full customer message becomes active.",
    "update_ticket": "Change priority, queue, status, and tags for the active ticket.",
    "reply_to_ticket": "Send a manual reply, or leave the message blank with auto-reply enabled to use Groq.",
    "add_internal_note": "Add an internal-only note for your support teammates.",
    "redact_sensitive_data": "Apply redaction before replying to sensitive customer content.",
    "complete_episode": "End the episode when you believe the work is done.",
}

MAIL_SYNC_INTERVAL_S = 30


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: #f6f8fb;
            color: #172033;
          }
          .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 3rem;
          }
          .hero-card,
          .soft-card,
          .ticket-shell,
          .detail-shell {
            border: 1px solid #dde6f2;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
          }
          .hero-card {
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
          }
          .soft-card {
            padding: 1rem 1.1rem;
          }
          .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #2563eb;
            margin-bottom: 0.35rem;
          }
          .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.02;
            letter-spacing: -0.04em;
            margin: 0 0 0.4rem 0;
          }
          .hero-copy,
          .muted-copy {
            color: #5f6f84;
            margin: 0;
          }
          .workflow-grid {
            display: grid;
            grid-template-columns: 1.1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
          }
          .guide-steps {
            margin: 0;
            padding-left: 1.1rem;
            color: #172033;
          }
          .guide-steps li {
            margin-bottom: 0.45rem;
          }
          .ticket-shell,
          .detail-shell {
            padding: 1rem 1.05rem;
            margin-bottom: 0.8rem;
          }
          .ticket-shell.active {
            border-color: rgba(37, 99, 235, 0.35);
            background: #f8fbff;
          }
          .ticket-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.35rem;
          }
          .ticket-title {
            font-weight: 700;
            font-size: 1rem;
            margin: 0;
          }
          .ticket-subject {
            margin: 0 0 0.55rem 0;
            font-size: 1.02rem;
            font-weight: 600;
            color: #172033;
          }
          .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.5rem;
          }
          .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 600;
            border: 1px solid #dbe4ef;
            background: #f4f7fb;
            color: #5f6f84;
          }
          .chip.live {
            background: #e8f8f3;
            border-color: rgba(15, 118, 110, 0.16);
            color: #0f766e;
          }
          .chip.alert {
            background: #fdeeee;
            border-color: rgba(220, 38, 38, 0.12);
            color: #dc2626;
          }
          .detail-shell p {
            white-space: pre-wrap;
            margin: 0.25rem 0 0;
          }
          .detail-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #5f6f84;
            margin-bottom: 0.35rem;
          }
          @media (max-width: 980px) {
            .workflow-grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_app_state() -> None:
    if "streamlit_env" not in st.session_state:
        env = SupportTriageEnvironment()
        st.session_state.streamlit_env = env
        st.session_state.latest_observation = env.reset()
        st.session_state.latest_grade = None
        st.session_state.latest_baseline = None
        st.session_state.flash_message = "Loaded the default support triage task."
        st.session_state.flash_kind = "success"
    st.session_state.setdefault("mail_auto_sync_enabled", True)
    st.session_state.setdefault("mail_last_sync_at", 0.0)


def set_flash(message: str, kind: str = "info") -> None:
    st.session_state.flash_message = message
    st.session_state.flash_kind = kind


def pop_flash() -> tuple[str | None, str]:
    message = st.session_state.pop("flash_message", None)
    kind = st.session_state.pop("flash_kind", "info")
    return message, kind


def current_env() -> SupportTriageEnvironment:
    ensure_app_state()
    return st.session_state.streamlit_env


def current_observation() -> SupportTriageObservation:
    ensure_app_state()
    return st.session_state.latest_observation


def reset_environment(payload: dict[str, Any]) -> None:
    env = current_env()
    st.session_state.latest_observation = env.reset(**payload)
    st.session_state.latest_grade = None
    set_flash(f"Loaded task {st.session_state.latest_observation.task_id}.", "success")


def run_action(action: SupportTriageAction) -> None:
    env = current_env()
    st.session_state.latest_observation = env.step(action)
    st.session_state.latest_grade = None
    set_flash(st.session_state.latest_observation.action_feedback, "success")


def grade_current_state() -> None:
    env = current_env()
    st.session_state.latest_grade = grade_support_episode(env.state)
    set_flash(f"Grader score: {st.session_state.latest_grade.score:.4f}", "success")


def run_baseline(agent_backend: str, model: str) -> None:
    st.session_state.latest_baseline = run_baseline_sync(
        agent_backend=agent_backend,
        model=model,
    )
    set_flash(
        f"Baseline complete with mean score {st.session_state.latest_baseline.mean_score:.4f}.",
        "success",
    )


def safe_text(value: str | None) -> str:
    return html.escape((value or "").strip() or "None yet.")


def chips_for_ticket(ticket: Any) -> str:
    values = [
        (ticket.priority, "alert" if ticket.priority == "urgent" else ""),
        (ticket.queue, ""),
        (ticket.status, ""),
        (ticket.customer_tier, ""),
        (ticket.channel, ""),
    ]
    if getattr(ticket, "opened", False):
        values.append(("opened", "live"))
    if getattr(ticket, "redaction_applied", False):
        values.append(("redacted", "live"))
    if getattr(ticket, "public_reply_sent", False):
        values.append(("replied", "live"))
    return "".join(
        f"<span class='chip {chip_class}'>{html.escape(str(label))}</span>"
        for label, chip_class in values
    )


def next_step_guidance(observation: SupportTriageObservation) -> tuple[str, str, str]:
    if observation.done:
        return (
            "Episode complete",
            "Review the outputs, inspect the grade, or reset into a new task.",
            "Done",
        )

    if observation.active_ticket is None:
        unopened = next((ticket for ticket in observation.inbox if not ticket.opened), None)
        if unopened:
            return (
                "Open a ticket",
                f"Start with {unopened.ticket_id} so the full customer message becomes active.",
                "Open",
            )
        return (
            "Inspect the queue",
            "Choose a ticket from the inbox to continue the workflow.",
            "Inbox",
        )

    active = observation.active_ticket
    if infer_sensitive_content(active.body) and not active.redaction_applied:
        return (
            "Redact before replying",
            f"{active.ticket_id} looks sensitive. Apply redaction before you send a public reply.",
            "Safety",
        )

    if not active.public_reply_sent:
        return (
            "Reply to the customer",
            f"Draft the next customer response for {active.ticket_id} with Groq or by hand.",
            "Reply",
        )

    return (
        "Grade or continue",
        "Inspect the grader, then keep working the inbox or complete the episode.",
        "Review",
    )


def render_flash() -> None:
    message, kind = pop_flash()
    if not message:
        return
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.info(message)


def schedule_browser_refresh(interval_s: int) -> None:
    components.html(
        f"""
        <script>
          window.setTimeout(function() {{
            window.parent.location.reload();
          }}, {int(interval_s * 1000)});
        </script>
        """,
        height=0,
        width=0,
    )


def sync_mailbox_if_needed(
    config: MailProviderConfig,
    *,
    force: bool = False,
    interval_s: int = MAIL_SYNC_INTERVAL_S,
) -> None:
    now = time.time()
    last_sync_at = float(st.session_state.get("mail_last_sync_at", 0.0))
    if not force and now - last_sync_at < interval_s:
        return

    items, created, message = sync_inbox_for_approval(config)
    st.session_state.mail_last_sync_at = now
    if created:
        set_flash(message, "success")
    elif force:
        set_flash(message, "info")


def render_sidebar() -> None:
    observation = current_observation()
    mail_config = load_mail_config_from_env()
    task_cards = public_task_cards()
    task_id_to_title = {
        card["task_id"]: f"{card['title']} ({card['task_id']})" for card in task_cards
    }

    with st.sidebar:
        st.markdown("### Streamlit Console")
        st.caption("This app uses the same environment and grader directly in-process.")

        if os.environ.get("GROQ_API_KEY"):
            st.success(f"Groq ready: {os.environ.get('GROQ_MODEL', DEFAULT_MODEL)}")
        else:
            st.warning("Groq not configured. Blank auto-replies will fail until GROQ_API_KEY is set.")

        st.markdown("### Reset Episode")
        selected_task = st.selectbox(
            "Task",
            options=[card["task_id"] for card in task_cards],
            format_func=lambda task_id: task_id_to_title[task_id],
            index=next(
                (idx for idx, card in enumerate(task_cards) if card["task_id"] == observation.task_id),
                0,
            ),
            key="sidebar_task_id",
        )

        is_custom = selected_task == CUSTOM_TASK_ID
        with st.form("reset_form", clear_on_submit=False):
            reset_payload: dict[str, Any] = {"task_id": selected_task}

            if is_custom:
                reset_payload["custom_ticket_id"] = st.text_input("Ticket ID", value="CUSTOM-4242")
                reset_payload["custom_subject"] = st.text_input(
                    "Subject",
                    value="Need help with my refund",
                )
                reset_payload["custom_body"] = st.text_area(
                    "Customer message",
                    value="Hi team, I was charged twice and need help understanding the refund status.",
                    height=150,
                )
                left, right = st.columns(2)
                reset_payload["custom_customer_tier"] = left.selectbox(
                    "Tier",
                    options=["starter", "pro", "enterprise", "vip"],
                    index=1,
                )
                reset_payload["custom_channel"] = right.selectbox(
                    "Channel",
                    options=["email", "chat"],
                    index=0,
                )
                reset_payload["custom_hours_open"] = st.number_input(
                    "Hours open",
                    min_value=0,
                    value=2,
                    step=1,
                )
                if st.checkbox("Contains sensitive data", value=False):
                    reset_payload["custom_contains_sensitive_data"] = True

            if st.form_submit_button("Reset Episode", use_container_width=True, type="primary"):
                try:
                    if is_custom and not str(reset_payload.get("custom_body", "")).strip():
                        raise ValueError("Custom sandbox requires a customer message.")
                    reset_environment(reset_payload)
                    st.rerun()
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    set_flash(str(exc), "error")
                    st.rerun()

        st.divider()
        st.markdown("### Baseline")
        baseline_agent = st.selectbox(
            "Agent backend",
            options=["auto", "scripted", "groq"],
            index=0,
            key="sidebar_baseline_agent",
        )
        baseline_model = st.text_input(
            "Groq model",
            value=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
            key="sidebar_baseline_model",
        )
        if st.button("Run Baseline", use_container_width=True):
            with st.spinner("Running baseline across the judged tasks..."):
                try:
                    run_baseline(baseline_agent, baseline_model.strip() or DEFAULT_MODEL)
                    st.rerun()
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    set_flash(str(exc), "error")
                    st.rerun()

        st.divider()
        st.markdown("### Button Usage")
        st.caption("Open Ticket: activate a queue item.")
        st.caption("Groq Reply: ask the backend to draft the customer response.")
        st.caption("Redact: remove risky content before replying.")
        st.caption("Grade Current State: score the current episode without ending it.")
        st.caption("Send Action: apply the manual action composer payload.")

        st.divider()
        st.markdown("### Live Mail Approval")
        if mail_config is None:
            st.warning("Set MAIL_EMAIL_ADDRESS and MAIL_APP_PASSWORD to enable real Gmail/Outlook approval.")
            st.caption("Optional env vars: MAIL_PROVIDER, MAIL_IMAP_HOST, MAIL_SMTP_HOST, MAIL_FOLDER.")
        else:
            st.success(f"Watching {mail_config.email_address} via {mail_config.provider}.")
            st.checkbox(
                "Auto-sync inbox every 30 seconds",
                key="mail_auto_sync_enabled",
            )
            if st.button("Sync inbox now", use_container_width=True):
                try:
                    sync_mailbox_if_needed(mail_config, force=True)
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    set_flash(str(exc), "error")
                st.rerun()
            st.caption("New inbound mail is drafted for approval and only sent after you click Send.")


def render_header(observation: SupportTriageObservation) -> None:
    next_title, next_detail, next_pill = next_step_guidance(observation)
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="eyebrow">{html.escape(observation.task_id)} • {html.escape(observation.difficulty)}</div>
          <div class="hero-title">{html.escape(observation.task_title)}</div>
          <p class="hero-copy">{html.escape(observation.objective)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="workflow-grid">
          <div class="soft-card">
            <div class="eyebrow">Next Step</div>
            <h3 style="margin:0 0 0.35rem 0;">{html.escape(next_title)}</h3>
            <p class="muted-copy">{html.escape(next_detail)}</p>
            <div class="chip-row" style="margin-top:0.8rem;">
              <span class="chip live">{html.escape(next_pill)}</span>
              <span class="chip">{html.escape(observation.action_feedback or "No feedback yet")}</span>
            </div>
          </div>
          <div class="soft-card">
            <div class="eyebrow">Quick Flow</div>
            <ol class="guide-steps">
              <li>Reset a judged task or your custom sandbox.</li>
              <li>Open a ticket from the inbox.</li>
              <li>Redact risky content before replying.</li>
              <li>Reply, note, update, or complete from the action composer.</li>
              <li>Grade the state whenever you want a deterministic score.</li>
            </ol>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("Reward", f"{float(observation.reward or 0.0):.4f}")
    metrics[1].metric("Completion", f"{float(observation.completion_score or 0.0):.4f}")
    metrics[2].metric("Steps Left", int(observation.remaining_steps or 0))
    metrics[3].metric("Invalid", int(observation.invalid_action_count or 0))


def render_inbox(observation: SupportTriageObservation) -> None:
    st.markdown("### Inbox")
    st.caption("Open a ticket to inspect the full customer message and make it active.")
    if not observation.inbox:
        st.info("Reset an episode to load tickets.")
        return

    for ticket in observation.inbox:
        is_active = observation.active_ticket is not None and observation.active_ticket.ticket_id == ticket.ticket_id
        st.markdown(
            f"""
            <div class="ticket-shell {'active' if is_active else ''}">
              <div class="ticket-heading">
                <p class="ticket-title">{html.escape(ticket.ticket_id)}</p>
                <div class="chip-row">{chips_for_ticket(ticket)}</div>
              </div>
              <p class="ticket-subject">{html.escape(ticket.subject)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Open {ticket.ticket_id}",
            key=f"open_ticket_{ticket.ticket_id}",
            use_container_width=True,
        ):
            run_action(SupportTriageAction(action_type="open_ticket", ticket_id=ticket.ticket_id))
            st.rerun()


def render_active_ticket(observation: SupportTriageObservation) -> None:
    st.markdown("### Active Ticket")
    st.caption("The current ticket shows the full customer message plus one-click actions.")

    active = observation.active_ticket
    if active is None:
        st.info("Open a ticket from the inbox to inspect it here.")
        return

    st.markdown(
        f"""
        <div class="detail-shell">
          <div class="eyebrow">Customer Message</div>
          <p>{safe_text(active.body)}</p>
        </div>
        <div class="detail-shell">
          <div class="eyebrow">Public Reply</div>
          <p>{safe_text(active.public_reply)}</p>
        </div>
        <div class="detail-shell">
          <div class="eyebrow">Internal Note</div>
          <p>{safe_text(active.internal_note)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_cols = st.columns(3)
    if quick_cols[0].button("Groq Reply", use_container_width=True):
        try:
            run_action(
                SupportTriageAction(
                    action_type="reply_to_ticket",
                    ticket_id=active.ticket_id,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            set_flash(str(exc), "error")
        st.rerun()

    if quick_cols[1].button("Redact", use_container_width=True):
        try:
            run_action(
                SupportTriageAction(
                    action_type="redact_sensitive_data",
                    ticket_id=active.ticket_id,
                    reason="Remove sensitive content before replying to the customer.",
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            set_flash(str(exc), "error")
        st.rerun()

    if quick_cols[2].button("Grade Current State", use_container_width=True):
        grade_current_state()
        st.rerun()


def render_mail_approval_queue(mail_config: MailProviderConfig | None) -> None:
    st.markdown("### Mail Approval Queue")
    st.caption("Real inbound emails create pending drafts here. Nothing is sent until you approve it.")

    if mail_config is None:
        st.info("Configure MAIL_EMAIL_ADDRESS and MAIL_APP_PASSWORD to enable the live Gmail/Outlook approval flow.")
        return

    items = pending_approval_items()
    if not items:
        st.info("No pending approval drafts yet. Sync the inbox or wait for new inbound mail.")
        return

    for item in items:
        with st.expander(f"{item.subject} • {item.from_address}", expanded=False):
            st.markdown(
                f"""
                <div class="detail-shell">
                  <div class="eyebrow">Incoming Email</div>
                  <p><strong>From:</strong> {html.escape(item.from_name or item.from_address)} &lt;{html.escape(item.from_address)}&gt;</p>
                  <p><strong>Received:</strong> {html.escape(item.received_at)}</p>
                  <p><strong>Snippet:</strong> {html.escape(item.snippet)}</p>
                  <p>{safe_text(item.body_text)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            widget_key = f"mail_{item.source_uid}"
            draft_value = st.text_area(
                "Draft reply",
                value=item.draft_reply,
                key=f"{widget_key}_draft",
                height=220,
            )
            action_cols = st.columns(3)
            if action_cols[0].button("Send approved reply", key=f"{widget_key}_send", type="primary", use_container_width=True):
                try:
                    sent_item = send_approved_reply(mail_config, item.id, draft_value)
                    set_flash(f"Sent approved reply to {sent_item.from_address}.", "success")
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    set_flash(str(exc), "error")
                st.rerun()

            if action_cols[1].button("Regenerate draft", key=f"{widget_key}_regen", use_container_width=True):
                try:
                    regenerate_approval_draft(item.id)
                    set_flash(f"Regenerated draft for {item.from_address}.", "success")
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    set_flash(str(exc), "error")
                st.rerun()

            if action_cols[2].button("Dismiss", key=f"{widget_key}_dismiss", use_container_width=True):
                try:
                    dismiss_approval_item(item.id)
                    set_flash(f"Dismissed draft for {item.from_address}.", "info")
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    set_flash(str(exc), "error")
                st.rerun()


def render_action_composer(observation: SupportTriageObservation) -> None:
    st.markdown("### Action Composer")
    st.caption("Use this when you want more control than the quick actions.")

    active_ticket_id = observation.active_ticket.ticket_id if observation.active_ticket else ""
    action_type = st.selectbox(
        "Action type",
        options=list(ACTION_HELP),
        key="composer_action_type",
    )
    st.caption(ACTION_HELP[action_type])

    with st.form("manual_action_form", clear_on_submit=False):
        ticket_id = ""
        if action_type != "complete_episode":
            ticket_id = st.text_input(
                "Ticket ID",
                value=active_ticket_id,
            )

        tags_input = st.text_input("Tags", value="")
        action_payload: dict[str, Any] = {
            "action_type": action_type,
            "tags": [tag.strip() for tag in tags_input.split(",") if tag.strip()],
        }

        if action_type == "update_ticket":
            left, middle, right = st.columns(3)
            action_payload["priority"] = left.selectbox(
                "Priority",
                options=["", *get_args(TicketPriority)],
                format_func=lambda value: value or "unchanged",
            )
            action_payload["queue"] = middle.selectbox(
                "Queue",
                options=["", *get_args(TicketQueue)],
                format_func=lambda value: value or "unchanged",
            )
            action_payload["status"] = right.selectbox(
                "Status",
                options=["", *get_args(TicketStatus)],
                format_func=lambda value: value or "unchanged",
            )

        if action_type == "reply_to_ticket":
            auto_reply = st.checkbox("Use Groq auto-reply when message is blank", value=True)
            message = st.text_area(
                "Reply message",
                value="",
                placeholder="Leave blank to let Groq draft the reply.",
                height=120,
            )
            if message.strip():
                action_payload["message"] = message.strip()
            elif not auto_reply:
                action_payload["message"] = ""

        if action_type == "add_internal_note":
            action_payload["note"] = st.text_area(
                "Internal note",
                value="",
                height=120,
            )

        if action_type == "redact_sensitive_data":
            action_payload["reason"] = st.text_area(
                "Redaction reason",
                value="Remove sensitive content before replying to the customer.",
                height=100,
            )

        if action_type == "complete_episode":
            action_payload["summary"] = st.text_area(
                "Completion summary",
                value=f"Finished task {observation.task_id}.",
                height=100,
            )

        submitted = st.form_submit_button("Send Action", use_container_width=True, type="primary")
        if submitted:
            try:
                if action_type != "complete_episode":
                    action_payload["ticket_id"] = ticket_id.strip()
                action = SupportTriageAction(**action_payload)
                run_action(action)
            except Exception as exc:  # pragma: no cover - surfaced in UI
                set_flash(str(exc), "error")
            st.rerun()


def render_outputs() -> None:
    env = current_env()
    observation = current_observation()
    tabs = st.tabs(["Feedback", "Grade", "Baseline", "Action History", "State JSON"])

    with tabs[0]:
        st.write(observation.action_feedback or "No feedback yet.")

    with tabs[1]:
        if st.session_state.latest_grade is None:
            st.info("Run Grade Current State to see the deterministic score breakdown.")
        else:
            st.json(st.session_state.latest_grade.model_dump())

    with tabs[2]:
        if st.session_state.latest_baseline is None:
            st.info("Run the baseline from the sidebar to compare the reference agent.")
        else:
            st.json(st.session_state.latest_baseline.model_dump())

    with tabs[3]:
        if not env.state.action_history:
            st.info("No actions recorded yet.")
        else:
            st.json([record.model_dump() for record in env.state.action_history])

    with tabs[4]:
        st.json(env.state.model_dump())


def main() -> None:
    st.set_page_config(
        page_title="Support Triage Streamlit Console",
        page_icon="📥",
        layout="wide",
    )
    inject_styles()
    ensure_app_state()
    mail_config = load_mail_config_from_env()
    if mail_config and st.session_state.get("mail_auto_sync_enabled", True):
        try:
            sync_mailbox_if_needed(mail_config, force=False)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            set_flash(str(exc), "error")
        schedule_browser_refresh(MAIL_SYNC_INTERVAL_S)

    render_sidebar()
    render_flash()

    observation = current_observation()
    render_header(observation)
    render_mail_approval_queue(mail_config)

    inbox_col, active_col = st.columns([1, 1], gap="large")
    with inbox_col:
        render_inbox(observation)
    with active_col:
        render_active_ticket(observation)

    render_action_composer(observation)
    render_outputs()


if __name__ == "__main__":
    main()
