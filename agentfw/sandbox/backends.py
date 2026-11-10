"""Backends for runner sessions: fakes, live backends, live seeding, success-check context.

* ``fake_backends`` / ``fake_servers``: a fresh, isolated set per call. ``id_map`` (a live run's
  seeded ids) makes the fakes answer for live ids, so an action-level injection's synthetic result
  (D-014: the mutated call goes to a fake, never the real API) is realistic for live arguments.
  ``sf_budget`` wraps the fake org in ``BudgetedSalesforce`` so an offline run exercises the
  Salesforce budget exactly as the live backend does (one charge per request).
* ``live_backends`` / ``seed_live``: the real sandbox APIs, seeded idempotently.
* ``check_context``: success checks over a session's backends; with the fake mail backend the
  Mailhog API is served from what it sent. ``session_mailhog`` hides mail that already existed when
  a live session started, so a fixed-recipient check cannot be satisfied by an earlier session.
* ``Allowlist``: the agent's view of the proxy, restricted to its allowlisted tools.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import orjson
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ListToolsResult

from agentfw.sandbox.agent import McpLike
from agentfw.sandbox.checks import CheckContext, mailhog_message_ids
from agentfw.sandbox.config import ApiBudget, BudgetExceeded, SandboxConfig
from agentfw.sandbox.proxy_py import error_result
from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tools import (
    mail_server,
    salesforce_server,
    slack_server,
    stripe_server,
    zendesk_server,
)
from agentfw.sandbox.tools.common import tool_error

__all__ = [
    "BUILDERS",
    "Allowlist",
    "BudgetedSalesforce",
    "check_context",
    "fake_backends",
    "fake_servers",
    "live_backends",
    "seed_live",
    "session_mailhog",
]

BUILDERS: dict[str, Callable[[Any], MCPServer]] = {
    "zendesk": zendesk_server.build_server,
    "stripe": stripe_server.build_server,
    "salesforce": salesforce_server.build_server,
    "slack": slack_server.build_server,
    "mail": mail_server.build_server,
}


class BudgetedSalesforce:
    """A fake org whose every request is charged to ``budget``, failing like the live backend."""

    is_fake = True

    def __init__(self, inner: salesforce_server.FakeSalesforceBackend, budget: ApiBudget) -> None:
        self._inner = inner
        self._budget = budget

    def _charge(self) -> None:
        try:
            self._budget.charge(1)
        except BudgetExceeded as exc:
            raise tool_error(f"budget exceeded: {exc}") from exc

    def run_soql(self, query: str) -> list[dict[str, Any]]:
        self._charge()
        return self._inner.run_soql(query)

    def get_record(self, obj: str, record_id: str) -> dict[str, Any]:
        self._charge()
        return self._inner.get_record(obj, record_id)

    def update_record(self, obj: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        self._charge()
        return self._inner.update_record(obj, record_id, fields)

    def mass_update_batch(self, obj: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._charge()
        return self._inner.mass_update_batch(obj, records)

    def create_task(self, what_id: str, subject: str) -> dict[str, Any]:
        self._charge()
        return self._inner.create_task(what_id, subject)

    # prompt-level planting writes through these (not API requests, so not charged)
    def ids(self) -> dict[str, str]:
        return self._inner.ids()

    def snapshot(self) -> Any:
        return self._inner.snapshot()

    def restore(self, snap: Any) -> None:
        self._inner.restore(snap)


def fake_backends(
    company: Company,
    channel: str,
    id_map: Mapping[str, str] | None = None,
    sf_budget: ApiBudget | None = None,
) -> dict[str, Any]:
    sf = salesforce_server.FakeSalesforceBackend(company, id_map)
    return {
        "zendesk": zendesk_server.FakeZendeskBackend(company, id_map),
        "stripe": stripe_server.FakeStripeBackend(company, id_map),
        "salesforce": sf if sf_budget is None else BudgetedSalesforce(sf, sf_budget),
        "slack": slack_server.FakeSlackBackend(channel),
        "mail": mail_server.FakeMailBackend(),
    }


def fake_servers(
    company: Company, channel: str, id_map: Mapping[str, str] | None = None
) -> dict[str, MCPServer]:
    return {n: BUILDERS[n](be) for n, be in fake_backends(company, channel, id_map).items()}


def _fake_mailhog(mail: mail_server.FakeMailBackend) -> Callable[[str], bytes]:
    """A Mailhog ``/api/v2/messages`` view of what the fake mail backend sent."""

    def fetch(url: str) -> bytes:
        items = []
        for n, m in enumerate(mail.sent):
            rcpts = []
            for addr in [*m["to"], *m["cc"]]:
                box, _, domain = str(addr).partition("@")
                rcpts.append({"Mailbox": box, "Domain": domain})
            items.append(
                {
                    "ID": f"fake-{n}",
                    "To": rcpts,
                    "Content": {"Headers": {"Subject": [m["subject"]]}},
                }
            )
        return orjson.dumps({"total": len(items), "items": items})

    return fetch


def session_mailhog(mailhog_http: str, fetch: Callable[[str], bytes]) -> Callable[[str], bytes]:
    """``fetch`` minus the messages Mailhog lists now (call it as a session starts)."""
    before = mailhog_message_ids(mailhog_http, fetch)

    def view(url: str) -> bytes:
        data = orjson.loads(fetch(url))
        items = [i for i in data.get("items", []) if str(i.get("ID")) not in before]
        return orjson.dumps({**data, "items": items, "count": len(items)})

    return view


def check_context(backends: Mapping[str, Any], mailhog_http: str) -> CheckContext:
    ctx = CheckContext(
        mailhog_http=mailhog_http,
        stripe=backends["stripe"],
        salesforce=backends["salesforce"],
        slack=backends["slack"],
        zendesk=backends["zendesk"],
    )
    mail = backends["mail"]
    if isinstance(mail, mail_server.FakeMailBackend):
        ctx.fetch = _fake_mailhog(mail)
    return ctx


def seed_live(
    cfg: SandboxConfig, company: Company, cache: Path, budget: ApiBudget
) -> dict[str, str]:
    """Idempotent seeding of the live org; the seeded ids are cached at ``cache``."""
    from agentfw.sandbox.seed.salesforce_seed import seed_salesforce
    from agentfw.sandbox.seed.stripe_seed import seed_stripe
    from agentfw.sandbox.seed.zendesk_seed import seed_zendesk

    if cache.exists():
        ids: dict[str, str] = orjson.loads(cache.read_bytes())
    else:
        ids = {**seed_stripe(cfg, company), **seed_salesforce(cfg, company, budget)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp")
        tmp.write_bytes(orjson.dumps(ids, option=orjson.OPT_SORT_KEYS))
        tmp.replace(cache)
    seed_zendesk(cfg.data_dir / zendesk_server.DB_NAME, company, ids)
    return ids


def live_backends(cfg: SandboxConfig, budget: ApiBudget) -> dict[str, Any]:
    return {
        "zendesk": zendesk_server.LiveZendeskBackend(cfg),
        "stripe": stripe_server.LiveStripeBackend(cfg),
        "salesforce": salesforce_server.LiveSalesforceBackend(cfg, budget=budget),
        "slack": slack_server.LiveSlackBackend.from_config(cfg),
        "mail": mail_server.LiveMailBackend(cfg),
    }


class Allowlist:
    """The agent's view of the proxy: only its allowlisted tools are listed or callable."""

    def __init__(self, inner: McpLike, allowed: Collection[str]) -> None:
        self._inner = inner
        self._allowed = frozenset(allowed)

    async def list_tools(self) -> ListToolsResult:
        res = await self._inner.list_tools()
        return ListToolsResult(tools=[t for t in res.tools if t.name in self._allowed])

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        if name not in self._allowed:
            return error_result(f"tool {name!r} is not allowed for this agent")
        return await self._inner.call_tool(name, arguments)
