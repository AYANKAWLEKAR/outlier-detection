"""Session backends: fakes keyed by a live id map, the per-session Mailhog view, budgeted SF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from agentfw.sandbox.backends import BudgetedSalesforce, fake_servers, session_mailhog
from agentfw.sandbox.checks import mailhog_has_message
from agentfw.sandbox.config import ApiBudget, UnsafeConfigError
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tools.salesforce_server import FakeSalesforceBackend

pytestmark = pytest.mark.anyio

COMPANY = build_company(0)
MAILHOG = "http://localhost:8025"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_fake_servers_answer_for_live_ids() -> None:
    ticket = COMPANY.tickets[0]
    acct = COMPANY.accounts[0]
    live_charge, live_acct = "ch_LIVE00000000000000000001", "001LIVE00000000AA"
    servers = fake_servers(
        COMPANY, "C0SANDBOX", {ticket.charge_key: live_charge, acct.key: live_acct}
    )
    async with Client(servers["stripe"]) as stripe:
        res = await stripe.call_tool("retrieve_charge", {"charge_id": live_charge})
        assert not res.is_error and res.structured_content is not None
        assert res.structured_content["id"] == live_charge
    async with Client(servers["salesforce"]) as sf:
        res = await sf.call_tool("get_record", {"object": "Account", "record_id": live_acct})
        assert not res.is_error
    async with Client(servers["zendesk"]) as zd:
        res = await zd.call_tool("get_ticket", {"ticket_id": ticket.key})
        assert res.structured_content is not None
        assert res.structured_content["charge_id"] == live_charge
    assert all(getattr(s, "is_fake", False) for s in servers.values())


def _item(mid: str, to: str, subject: str) -> dict[str, Any]:
    box, _, domain = to.partition("@")
    return {
        "ID": mid,
        "To": [{"Mailbox": box, "Domain": domain}],
        "Content": {"Headers": {"Subject": [subject]}},
    }


def test_session_mailhog_hides_mail_sent_before_the_session() -> None:
    store: dict[str, Any] = {"items": [_item("m1", "a@x.example", "Hello")]}

    def fetch(url: str) -> bytes:
        assert url == f"{MAILHOG}/api/v2/messages"
        return orjson.dumps(store)

    view = session_mailhog(MAILHOG, fetch)
    assert not mailhog_has_message(MAILHOG, "a@x.example", "Hello", view)
    store["items"].append(_item("m2", "a@x.example", "Hello again"))
    assert mailhog_has_message(MAILHOG, "a@x.example", "Hello", view)
    with pytest.raises(UnsafeConfigError):
        session_mailhog("http://mail.example.com:8025", fetch)


def test_budgeted_salesforce_charges_each_request_and_fails_like_live(tmp_path: Path) -> None:
    budget = ApiBudget(tmp_path / "sf.json", 2)
    sf = BudgetedSalesforce(FakeSalesforceBackend(COMPANY), budget)
    rid = sf.ids()[COMPANY.accounts[0].key]
    sf.get_record("Account", rid)
    sf.update_record("Account", rid, {"Description": "x"})
    assert budget.remaining() == 0
    with pytest.raises(ToolError, match="budget exceeded"):
        sf.get_record("Account", rid)
    snap = sf.snapshot()  # prompt planting still works through the wrapper
    sf.restore(snap)
