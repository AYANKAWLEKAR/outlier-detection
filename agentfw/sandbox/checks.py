"""Success checks: evaluators over backend Protocols (fakes in tests, live backends later)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

import orjson
from mcp.server.mcpserver.exceptions import ToolError

from agentfw.sandbox.config import ALLOWED_SMTP_HOSTS, UnsafeConfigError
from agentfw.sandbox.tools.salesforce_server import SalesforceBackend
from agentfw.sandbox.tools.slack_server import SlackBackend
from agentfw.sandbox.tools.stripe_server import StripeBackend
from agentfw.sandbox.tools.zendesk_server import ZendeskBackend


def _fetch(url: str) -> bytes:
    with urlopen(url, timeout=10) as res:  # noqa: S310 - scheme and host validated by caller
        data: bytes = res.read()
    return data


def _local_http(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in ALLOWED_SMTP_HOSTS:
        raise UnsafeConfigError(f"mailhog_http {url!r} refused: local Mailhog only")
    return url.rstrip("/")


def mailhog_has_message(
    mailhog_http: str,
    to: str,
    subject_contains: str,
    fetch: Callable[[str], bytes] = _fetch,
) -> bool:
    base = _local_http(mailhog_http)
    data = orjson.loads(fetch(f"{base}/api/v2/messages"))
    want = to.lower()
    needle = subject_contains.lower()
    for item in data.get("items", []):
        rcpts = {f"{t['Mailbox']}@{t['Domain']}".lower() for t in item.get("To", [])}
        subjects = item.get("Content", {}).get("Headers", {}).get("Subject", [])
        if want in rcpts and any(needle in s.lower() for s in subjects):
            return True
    return False


def stripe_refund_exists(stripe: StripeBackend, charge_id: str) -> bool:
    try:
        charge = stripe.retrieve_charge(charge_id)
    except ToolError:
        return False
    return float(charge.get("amount_refunded", 0)) > 0


def _same(actual: Any, expected: Any) -> bool:
    """Equality that also treats numbers and their string forms alike (`33770` == `"33770.0"`)."""
    if actual == expected:
        return True
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def sf_field_equals(
    sf: SalesforceBackend, obj: str, record_id: str, field_: str, value: Any
) -> bool:
    try:
        return _same(sf.get_record(obj, record_id).get(field_), value)
    except ToolError:
        return False


def slack_contains(slack: SlackBackend, text: str, limit: int = 100) -> bool:
    return any(text in str(m.get("text", "")) for m in slack.read_channel(limit))


def zendesk_status(zendesk: ZendeskBackend, ticket_id: str, status: str) -> bool:
    try:
        return bool(zendesk.get_ticket(ticket_id)["status"] == status)
    except ToolError:
        return False


@dataclass(slots=True)
class CheckContext:
    mailhog_http: str = "http://localhost:8025"
    stripe: StripeBackend | None = None
    salesforce: SalesforceBackend | None = None
    slack: SlackBackend | None = None
    zendesk: ZendeskBackend | None = None
    fetch: Callable[[str], bytes] = _fetch


def _need[T](backend: T | None, name: str) -> T:
    if backend is None:
        raise ValueError(f"check needs a {name} backend in the CheckContext")
    return backend


@dataclass(frozen=True, slots=True)
class SuccessCheck:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, ctx: CheckContext) -> bool:
        p = self.params
        match self.kind:
            case "mailhog_has_message":
                return mailhog_has_message(
                    ctx.mailhog_http, p["to"], p["subject_contains"], ctx.fetch
                )
            case "stripe_refund_exists":
                return stripe_refund_exists(_need(ctx.stripe, "stripe"), p["charge_id"])
            case "sf_field_equals":
                return sf_field_equals(
                    _need(ctx.salesforce, "salesforce"),
                    p["object"],
                    p["record_id"],
                    p["field"],
                    p["value"],
                )
            case "slack_contains":
                return slack_contains(_need(ctx.slack, "slack"), p["text"])
            case "zendesk_status":
                return zendesk_status(_need(ctx.zendesk, "zendesk"), p["ticket_id"], p["status"])
            case _:
                raise ValueError(f"unknown success check kind {self.kind!r}")
