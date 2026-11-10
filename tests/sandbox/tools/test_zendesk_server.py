"""Zendesk (local SQLite) MCP server: round-trips through an in-memory MCP client."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from agentfw.core.patterns import is_id_like
from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.seed.zendesk_seed import seed_zendesk
from agentfw.sandbox.tools.stripe_server import fake_stripe_ids
from agentfw.sandbox.tools.zendesk_server import (
    FakeZendeskBackend,
    LiveZendeskBackend,
    build_server,
)
from tests.sandbox.tools.conftest import call, tool_names

pytestmark = pytest.mark.anyio

COMPANY = build_company(0)
FIRST = COMPANY.tickets[0]


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    return build_server(FakeZendeskBackend(COMPANY))


async def test_tool_names(server) -> None:  # type: ignore[no-untyped-def]
    assert await tool_names(server) == {
        "get_ticket",
        "list_tickets",
        "add_comment",
        "update_status",
    }


async def test_get_ticket_round_trip(server) -> None:  # type: ignore[no-untyped-def]
    err, t = await call(server, "get_ticket", ticket_id=FIRST.key)
    assert not err
    assert t["id"] == FIRST.key
    assert t["status"] == "open"
    assert t["customer_email"] == FIRST.customer_email
    assert t["charge_id"] == fake_stripe_ids(COMPANY)[FIRST.charge_key]
    assert t["comments"] == []


async def test_list_tickets_wrapped_and_limited(server) -> None:  # type: ignore[no-untyped-def]
    err, res = await call(server, "list_tickets")
    assert not err
    assert set(res) == {"data"}
    assert 0 < len(res["data"]) <= 20
    assert {t["status"] for t in res["data"]} == {"open"}
    err, res = await call(server, "list_tickets", limit=2)
    assert len(res["data"]) == 2


async def test_add_comment_then_visible(server) -> None:  # type: ignore[no-untyped-def]
    err, c = await call(server, "add_comment", ticket_id=FIRST.key, body="Looking into it.")
    assert not err
    assert c["ticket_id"] == FIRST.key
    _, t = await call(server, "get_ticket", ticket_id=FIRST.key)
    assert [x["body"] for x in t["comments"]] == ["Looking into it."]
    assert t["comments"][0]["id"] == c["id"]


async def test_update_status_moves_between_lists(server) -> None:  # type: ignore[no-untyped-def]
    err, t = await call(server, "update_status", ticket_id=FIRST.key, status="solved")
    assert not err
    assert t["status"] == "solved"
    _, solved = await call(server, "list_tickets", status="solved")
    assert [x["id"] for x in solved["data"]] == [FIRST.key]
    _, opened = await call(server, "list_tickets", limit=100)
    assert FIRST.key not in [x["id"] for x in opened["data"]]


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("get_ticket", {}),
        ("add_comment", {"body": "x"}),
        ("update_status", {"status": "solved"}),
    ],
)
async def test_unknown_ticket_is_error(server, tool, args) -> None:  # type: ignore[no-untyped-def]
    err, msg = await call(server, tool, ticket_id="tk_NOSUCHTICKET", **args)
    assert err
    assert "tk_NOSUCHTICKET" in msg


async def test_invalid_status_is_error(server) -> None:  # type: ignore[no-untyped-def]
    err, msg = await call(server, "update_status", ticket_id=FIRST.key, status="exploded")
    assert err
    assert "status" in msg
    err, _ = await call(server, "list_tickets", status="exploded")
    assert err


async def test_fake_backends_are_isolated() -> None:
    a, b = FakeZendeskBackend(COMPANY), FakeZendeskBackend(COMPANY)
    a.update_status(FIRST.key, "solved")
    assert b.get_ticket(FIRST.key)["status"] == "open"


async def test_live_backend_uses_file_db_under_data_dir(tmp_path: Path) -> None:
    cfg = SandboxConfig(
        llm_base_url="https://llm.example/v1",
        llm_api_key=SecretStr("x"),
        llm_model="m",
        judge_base_url="https://j.example/v1",
        judge_api_key=SecretStr("x"),
        judge_model="m",
        stripe_secret_key=SecretStr("sk_test_fake"),
        sf_username="u@x.example",
        sf_password=SecretStr("x"),
        sf_security_token=SecretStr("x"),
        slack_bot_token=SecretStr("x"),
        slack_channel_id="C1",
        llm_usd_budget=1.0,
        price_in_per_m=1.0,
        price_out_per_m=1.0,
        data_dir=tmp_path,
    )
    seed_zendesk(tmp_path / "zendesk.db", COMPANY, fake_stripe_ids(COMPANY))
    backend = LiveZendeskBackend(cfg)
    srv = build_server(backend)
    err, _ = await call(srv, "add_comment", ticket_id=FIRST.key, body="persisted")
    assert not err
    # A second backend on the same file sees the write (durable, unlike the fake).
    assert LiveZendeskBackend(cfg).get_ticket(FIRST.key)["comments"][0]["body"] == "persisted"


async def test_all_ids_returned_are_id_like(server) -> None:  # type: ignore[no-untyped-def]
    _, res = await call(server, "list_tickets", limit=100)
    assert res["data"]
    for t in res["data"]:
        assert t["id"].startswith("tk_") and len(t["id"]) >= 13
        assert t["id"][3:] == t["id"][3:].upper()
        assert is_id_like(t["id"])
        assert is_id_like(t["customer_id"])
        assert is_id_like(t["charge_id"])
    assert len({k.key for k in COMPANY.tickets}) == len(COMPANY.tickets)
    assert all(is_id_like(k.key) for k in COMPANY.tickets)
