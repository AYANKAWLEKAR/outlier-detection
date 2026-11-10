"""Mail MCP server: email only ever goes to Mailhog (localhost / compose), never a real relay."""

from __future__ import annotations

import re
import smtplib
import threading
from email.message import EmailMessage
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer

from agentfw.sandbox.config import ALLOWED_SMTP_HOSTS, SandboxConfig, UnsafeConfigError
from agentfw.sandbox.tools.common import mark_fake, tool_error

_ADDR = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")
_SENDER = "agent@sandbox.example"


class MailBackend(Protocol):
    def send_email(
        self, to: list[str], subject: str, body: str, cc: list[str]
    ) -> dict[str, Any]: ...


def _validate(to: list[str], subject: str, cc: list[str]) -> None:
    if not to:
        raise tool_error("to must contain at least one recipient")
    for a in [*to, *cc]:
        if _ADDR.match(a) is None:
            raise tool_error(f"invalid email address {a!r}")
    if "\r" in subject or "\n" in subject:
        raise tool_error("subject must not contain line breaks")


class FakeMailBackend:
    is_fake = True

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def send_email(self, to: list[str], subject: str, body: str, cc: list[str]) -> dict[str, Any]:
        _validate(to, subject, cc)
        with self._lock:
            self.sent.append({"to": list(to), "cc": list(cc), "subject": subject, "body": body})
        return {"status": "sent", "recipients": len(to) + len(cc)}


class LiveMailBackend:
    """smtplib to the Mailhog SMTP port configured in `SandboxConfig` (local hosts only)."""

    def __init__(self, cfg: SandboxConfig) -> None:
        if cfg.smtp_host not in ALLOWED_SMTP_HOSTS:
            raise UnsafeConfigError(f"smtp_host {cfg.smtp_host!r} refused: Mailhog only")
        self._host, self._port = cfg.smtp_host, cfg.smtp_port

    def send_email(self, to: list[str], subject: str, body: str, cc: list[str]) -> dict[str, Any]:
        _validate(to, subject, cc)
        msg = EmailMessage()
        msg["From"] = _SENDER
        msg["To"] = ", ".join(to)
        if cc:
            msg["Cc"] = ", ".join(cc)
        msg["Subject"] = subject
        msg.set_content(body)
        try:
            with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
                smtp.send_message(msg, to_addrs=[*to, *cc])
        except (OSError, smtplib.SMTPException) as exc:
            raise tool_error(f"mail delivery failed: {exc}") from exc
        return {"status": "sent", "recipients": len(to) + len(cc)}


def build_server(backend: MailBackend) -> MCPServer:
    server = MCPServer("mail")

    @server.tool()
    def send_email(
        to: list[str], subject: str, body: str, cc: list[str] | None = None
    ) -> dict[str, Any]:
        """Send an email."""
        return backend.send_email(to, subject, body, cc or [])

    return mark_fake(server, backend)
