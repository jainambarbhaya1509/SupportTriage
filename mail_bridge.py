"""Mailbox polling and human-approval draft workflow for real email providers."""

from __future__ import annotations

import json
import os
import re
import ssl
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import formataddr, parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
import imaplib
from pathlib import Path
import smtplib
from typing import Any

from llm_client import chat_completion, default_model_name, resolve_openai_config
from openai_reply import clean_model_text

PROVIDER_DEFAULTS = {
    "gmail": {
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
    },
    "outlook": {
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "smtp_host": "smtp-mail.outlook.com",
        "smtp_port": 587,
    },
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def as_text(self) -> str:
        return "\n".join(self.parts)


@dataclass
class MailProviderConfig:
    provider: str
    email_address: str
    password: str
    imap_host: str
    imap_port: int
    smtp_host: str
    smtp_port: int
    folder: str = "INBOX"
    poll_limit: int = 20


@dataclass
class ApprovalItem:
    id: str
    source_uid: int
    message_id: str | None
    provider: str
    from_name: str
    from_address: str
    subject: str
    body_text: str
    snippet: str
    received_at: str
    status: str
    draft_reply: str
    in_reply_to: str | None = None
    references: str | None = None
    sent_at: str | None = None
    last_error: str | None = None


def approval_state_path() -> Path:
    path = os.environ.get("MAIL_APPROVAL_STATE_PATH", "outputs/mail_approval_state.json")
    return Path(path)


def load_mail_config_from_env() -> MailProviderConfig | None:
    provider = os.environ.get("MAIL_PROVIDER", "gmail").strip().lower()
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["gmail"])
    email_address = os.environ.get("MAIL_EMAIL_ADDRESS", "").strip()
    password = os.environ.get("MAIL_APP_PASSWORD") or os.environ.get("MAIL_PASSWORD") or ""
    if not email_address or not password:
        return None

    return MailProviderConfig(
        provider=provider,
        email_address=email_address,
        password=password,
        imap_host=os.environ.get("MAIL_IMAP_HOST", defaults["imap_host"]),
        imap_port=int(os.environ.get("MAIL_IMAP_PORT", defaults["imap_port"])),
        smtp_host=os.environ.get("MAIL_SMTP_HOST", defaults["smtp_host"]),
        smtp_port=int(os.environ.get("MAIL_SMTP_PORT", defaults["smtp_port"])),
        folder=os.environ.get("MAIL_FOLDER", "INBOX"),
        poll_limit=int(os.environ.get("MAIL_POLL_LIMIT", "20")),
    )


def load_mail_approval_state() -> dict[str, Any]:
    path = approval_state_path()
    if not path.exists():
        return {"meta": {}, "items": []}
    return json.loads(path.read_text())


def save_mail_approval_state(state: dict[str, Any]) -> None:
    path = approval_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def approval_items(state: dict[str, Any]) -> list[ApprovalItem]:
    return [ApprovalItem(**item) for item in state.get("items", [])]


def pending_approval_items() -> list[ApprovalItem]:
    state = load_mail_approval_state()
    return [item for item in approval_items(state) if item.status == "pending_approval"]


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value))).strip()


def _html_to_text(raw_html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    return parser.as_text()


def _message_plaintext(message: Message) -> str:
    if message.is_multipart():
        plain_parts: list[str] = []
        html_parts: list[str] = []
        for part in message.walk():
            disposition = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disposition:
                continue
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace").strip()
            if part.get_content_type() == "text/plain" and text:
                plain_parts.append(text)
            elif part.get_content_type() == "text/html" and text:
                html_parts.append(_html_to_text(text))
        if plain_parts:
            return "\n\n".join(plain_parts).strip()
        if html_parts:
            return "\n\n".join(html_parts).strip()
        return ""

    payload = message.get_payload(decode=True) or b""
    charset = message.get_content_charset() or "utf-8"
    text = payload.decode(charset, errors="replace").strip()
    if message.get_content_type() == "text/html":
        return _html_to_text(text)
    return text


def _received_at(message: Message) -> str:
    raw_date = message.get("Date")
    if not raw_date:
        return datetime.now(timezone.utc).isoformat()
    try:
        return parsedate_to_datetime(raw_date).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()


def _snippet(text: str, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _subject_for_reply(subject: str) -> str:
    if subject.lower().startswith("re:"):
        return subject
    return f"Re: {subject}"


def _fallback_reply(item: ApprovalItem) -> str:
    first_name = item.from_name.split(" ", 1)[0] if item.from_name else "there"
    return (
        f"Hi {first_name},\n\n"
        "Thanks for reaching out. I reviewed your message and drafted a response for approval. "
        "I will follow up shortly with the next step.\n\n"
        "Best,\n"
        "Jainam"
    )


def _generate_mail_draft(item: ApprovalItem) -> str:
    config = resolve_openai_config(model_override=default_model_name())
    if config is None:
        return _fallback_reply(item)

    prompt = (
        "You are drafting a customer support email reply for human approval.\n"
        "Write a concise, professional response in plain text.\n"
        "Do not use markdown.\n"
        "Do not promise refunds, fixes, or outcomes that are not explicitly confirmed.\n"
        "Acknowledge the issue, reflect the user's concern, and describe the next safe step.\n"
        f"Sender: {item.from_name or item.from_address}\n"
        f"Subject: {item.subject}\n"
        f"Email body:\n{item.body_text}\n"
    )
    raw_text = chat_completion(
        config,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=250,
    )
    cleaned = clean_model_text(raw_text)
    return cleaned or _fallback_reply(item)


def sync_inbox_for_approval(config: MailProviderConfig) -> tuple[list[ApprovalItem], int, str]:
    state = load_mail_approval_state()
    items = approval_items(state)
    highest_uid = int(state.get("meta", {}).get("highest_uid", 0))

    with imaplib.IMAP4_SSL(config.imap_host, config.imap_port) as mailbox:
        mailbox.login(config.email_address, config.password)
        mailbox.select(config.folder)
        status, data = mailbox.uid("search", None, "ALL")
        if status != "OK":
            raise RuntimeError("Failed to search mailbox.")

        all_uids = [int(uid) for uid in (data[0] or b"").split() if uid]
        if not all_uids:
            state["meta"] = {"highest_uid": highest_uid}
            save_mail_approval_state(state)
            return items, 0, "No mail found in the selected folder."

        latest_uid = max(all_uids)
        if highest_uid == 0:
            state["meta"] = {"highest_uid": latest_uid}
            save_mail_approval_state(state)
            return items, 0, "Initialized mailbox watch. New mail from now on will generate approval drafts."

        new_uids = [uid for uid in all_uids if uid > highest_uid][-config.poll_limit :]
        created = 0
        existing_keys = {item.message_id or f"uid:{item.source_uid}" for item in items}

        for uid in new_uids:
            status, fetched = mailbox.uid("fetch", str(uid), "(BODY.PEEK[])")
            if status != "OK":
                continue

            raw_message = b""
            for part in fetched:
                if isinstance(part, tuple):
                    raw_message = part[1]
                    break
            if not raw_message:
                continue

            message = message_from_bytes(raw_message)
            from_name, from_address = parseaddr(message.get("From", ""))
            if from_address.lower() == config.email_address.lower():
                continue

            message_id = _decode_header_value(message.get("Message-ID"))
            identity = message_id or f"uid:{uid}"
            if identity in existing_keys:
                continue

            body_text = _message_plaintext(message)
            item = ApprovalItem(
                id=identity,
                source_uid=uid,
                message_id=message_id or None,
                provider=config.provider,
                from_name=_decode_header_value(from_name),
                from_address=from_address,
                subject=_decode_header_value(message.get("Subject")) or "(no subject)",
                body_text=body_text,
                snippet=_snippet(body_text),
                received_at=_received_at(message),
                status="pending_approval",
                draft_reply="",
                in_reply_to=_decode_header_value(message.get("In-Reply-To")) or None,
                references=_decode_header_value(message.get("References")) or None,
            )
            item.draft_reply = _generate_mail_draft(item)
            items.insert(0, item)
            existing_keys.add(identity)
            created += 1

    state["meta"] = {"highest_uid": latest_uid}
    state["items"] = [asdict(item) for item in items]
    save_mail_approval_state(state)
    if created:
        return items, created, f"Created {created} approval draft(s) from new inbound email."
    return items, 0, "No new inbound email needed approval drafts."


def _replace_item(updated_item: ApprovalItem) -> None:
    state = load_mail_approval_state()
    items = approval_items(state)
    state["items"] = [
        asdict(updated_item if item.id == updated_item.id else item)
        for item in items
    ]
    save_mail_approval_state(state)


def regenerate_approval_draft(item_id: str) -> ApprovalItem:
    items = pending_approval_items()
    for item in items:
        if item.id == item_id:
            item.draft_reply = _generate_mail_draft(item)
            item.last_error = None
            _replace_item(item)
            return item
    raise KeyError(f"Approval item {item_id} was not found.")


def dismiss_approval_item(item_id: str) -> ApprovalItem:
    state = load_mail_approval_state()
    items = approval_items(state)
    for item in items:
        if item.id == item_id:
            item.status = "dismissed"
            item.last_error = None
            _replace_item(item)
            return item
    raise KeyError(f"Approval item {item_id} was not found.")


def send_approved_reply(
    config: MailProviderConfig,
    item_id: str,
    approved_reply: str,
) -> ApprovalItem:
    state = load_mail_approval_state()
    items = approval_items(state)
    item = next((entry for entry in items if entry.id == item_id), None)
    if item is None:
        raise KeyError(f"Approval item {item_id} was not found.")

    message = EmailMessage()
    message["To"] = formataddr((item.from_name, item.from_address)) if item.from_name else item.from_address
    message["From"] = config.email_address
    message["Subject"] = _subject_for_reply(item.subject)
    if item.message_id:
        message["In-Reply-To"] = item.message_id
    references = " ".join(
        ref for ref in [item.references, item.message_id] if ref
    ).strip()
    if references:
        message["References"] = references
    message.set_content(approved_reply.strip())

    context = ssl.create_default_context()
    with smtplib.SMTP(config.smtp_host, config.smtp_port) as smtp:
        smtp.starttls(context=context)
        smtp.login(config.email_address, config.password)
        smtp.send_message(message)

    item.draft_reply = approved_reply.strip()
    item.status = "sent"
    item.sent_at = datetime.now(timezone.utc).isoformat()
    item.last_error = None
    _replace_item(item)
    return item
