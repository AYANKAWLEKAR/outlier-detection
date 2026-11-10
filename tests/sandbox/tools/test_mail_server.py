"""Mail MCP server: Fake records messages; live goes only to Mailhog."""

from __future__ import annotations

import os
from typing import Any

import orjson
import pytest

from agentfw.sandbox.config import SandboxConfig, UnsafeConfigError
from agentfw.sandbox.tools.mail_server import FakeMailBackend, LiveMailBackend, build_server
from tests.sandbox.tools.conftest import call, tool_names


@pytest.fixture
def backend() -> FakeMailBackend:
    return FakeMailBackend()


@pytest.mark.anyio
async def test_tool_names(backend) -> None:  # type: ignore[no-untyped-def]
    assert await tool_names(build_server(backend)) == {"send_email"}


@pytest.mark.anyio
async def test_send_email_records(backend) -> None:  # type: ignore[no-untyped-def]
    srv = build_server(backend)
    err, res = await call(
        srv,
        "send_email",
        to=["a@acme.example"],
        subject="Hi",
        body="Hello",
        cc=["b@acme.example"],
    )
    assert not err and res["status"] == "sent" and res["recipients"] == 2
    assert backend.sent[0]["to"] == ["a@acme.example"]
    assert backend.sent[0]["cc"] == ["b@acme.example"]
    assert backend.sent[0]["subject"] == "Hi"


@pytest.mark.anyio
async def test_send_email_validation(backend) -> None:  # type: ignore[no-untyped-def]
    srv = build_server(backend)
    err, _ = await call(srv, "send_email", to=[], subject="s", body="b")
    assert err
    err, _ = await call(srv, "send_email", to=["not-an-address"], subject="s", body="b")
    assert err
    err, _ = await call(
        srv, "send_email", to=["a@acme.example"], subject="s\r\nBcc: x@y.z", body="b"
    )
    assert err
    assert backend.sent == []


def _cfg(**over: Any) -> SandboxConfig:
    env = {n.upper(): "x" for n in SandboxConfig.REQUIRED_ENV}
    env.update(
        STRIPE_SECRET_KEY="sk_test_abc", LLM_USD_BUDGET="1", PRICE_IN_PER_M="1", PRICE_OUT_PER_M="1"
    )
    env.update({k.upper(): str(v) for k, v in over.items()})
    return SandboxConfig.from_env(env)


def test_config_refuses_non_local_smtp() -> None:
    with pytest.raises(UnsafeConfigError):
        _cfg(smtp_host="smtp.gmail.com")


@pytest.mark.anyio
async def test_live_backend_uses_smtp_to_configured_host(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, Any] = {}

    class FakeSMTP:
        def __init__(self, host: str, port: int, timeout: float = 0) -> None:
            seen["addr"] = (host, port)

        def __enter__(self) -> FakeSMTP:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def send_message(self, msg: Any, to_addrs: list[str]) -> None:
            seen["to"] = to_addrs
            seen["subject"] = msg["Subject"]

    monkeypatch.setattr("agentfw.sandbox.tools.mail_server.smtplib.SMTP", FakeSMTP)
    srv = build_server(LiveMailBackend(_cfg(smtp_host="mailhog", smtp_port=1025)))
    err, _ = await call(
        srv, "send_email", to=["a@acme.example"], subject="S", body="B", cc=["c@acme.example"]
    )
    assert not err
    assert seen["addr"] == ("mailhog", 1025)
    assert seen["to"] == ["a@acme.example", "c@acme.example"] and seen["subject"] == "S"


def test_live_backend_refuses_non_local_host() -> None:
    cfg = _cfg()
    object.__setattr__(cfg, "smtp_host", "smtp.example.com")  # bypass validator on purpose
    with pytest.raises(UnsafeConfigError):
        LiveMailBackend(cfg)


@pytest.mark.live
@pytest.mark.anyio
async def test_live_send_email_shows_in_mailhog() -> None:
    from urllib.request import urlopen

    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    cfg = SandboxConfig.from_env()
    srv = build_server(LiveMailBackend(cfg))
    err, _ = await call(
        srv, "send_email", to=["live@acme.example"], subject="afw-live-test", body="b"
    )
    assert not err
    with urlopen(f"{cfg.mailhog_http}/api/v2/messages", timeout=10) as r:  # noqa: S310
        assert b"afw-live-test" in orjson.dumps(orjson.loads(r.read()))
