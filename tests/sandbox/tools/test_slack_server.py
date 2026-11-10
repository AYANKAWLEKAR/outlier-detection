"""Slack MCP server: only the one configured sandbox channel is ever touched."""

from __future__ import annotations

import os
from typing import Any

import pytest

from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.tools.slack_server import FakeSlackBackend, LiveSlackBackend, build_server
from tests.sandbox.tools.conftest import call, tool_names

CHAN = "C0SANDBOX1"


@pytest.fixture
def backend() -> FakeSlackBackend:
    return FakeSlackBackend(CHAN)


@pytest.mark.anyio
async def test_tool_names(backend) -> None:  # type: ignore[no-untyped-def]
    assert await tool_names(build_server(backend)) == {"post_message", "read_channel"}


@pytest.mark.anyio
async def test_post_then_read(backend) -> None:  # type: ignore[no-untyped-def]
    srv = build_server(backend)
    err, res = await call(srv, "post_message", channel=CHAN, text="hello there")
    assert not err and res["ok"] is True and res["channel"] == CHAN
    err, msgs = await call(srv, "read_channel", limit=5)
    assert not err and [m["text"] for m in msgs["data"]] == ["hello there"]


@pytest.mark.anyio
async def test_other_channel_rejected(backend) -> None:  # type: ignore[no-untyped-def]
    srv = build_server(backend)
    err, msg = await call(srv, "post_message", channel="C0OTHER", text="nope")
    assert err and "channel" in msg.lower()
    err, msgs = await call(srv, "read_channel")
    assert msgs["data"] == []


@pytest.mark.anyio
async def test_read_limit_and_order(backend) -> None:  # type: ignore[no-untyped-def]
    srv = build_server(backend)
    for i in range(5):
        await call(srv, "post_message", channel=CHAN, text=f"m{i}")
    _, msgs = await call(srv, "read_channel", limit=2)
    assert [m["text"] for m in msgs["data"]] == ["m4", "m3"]  # newest first, like Slack
    err, _ = await call(srv, "read_channel", limit=0)
    assert err


@pytest.mark.anyio
async def test_live_backend_rejects_other_channel_without_calling_api() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def chat_postMessage(self, **kw: Any) -> dict[str, Any]:  # noqa: N802
            self.calls += 1
            return {"ok": True, "ts": "1.0", "channel": kw["channel"]}

    client = Client()
    srv = build_server(LiveSlackBackend(CHAN, client=client))
    err, _ = await call(srv, "post_message", channel="C0OTHER", text="x")
    assert err and client.calls == 0
    err, _ = await call(srv, "post_message", channel=CHAN, text="x")
    assert not err and client.calls == 1


@pytest.mark.live
@pytest.mark.anyio
async def test_live_post_visible_via_read_channel() -> None:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    cfg = SandboxConfig.from_env()
    srv = build_server(LiveSlackBackend.from_config(cfg))
    err, _ = await call(srv, "post_message", channel=cfg.slack_channel_id, text="afw-live-test")
    assert not err
    _, msgs = await call(srv, "read_channel", limit=10)
    assert "afw-live-test" in [m["text"] for m in msgs["data"]]
