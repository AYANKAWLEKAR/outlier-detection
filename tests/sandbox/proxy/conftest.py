"""Proxy test helpers: in-process downstream servers, a memory sink, a spying downstream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from mcp import Client
from mcp.types import CallToolResult, ListToolsResult

from agentfw.bench.protocol import Detector
from agentfw.core.envelope import DetectorResult, Label, RawCall
from agentfw.sandbox.proxy_py import ActionMutator, FirewallProxy, SessionInfo
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tools import stripe_server, zendesk_server
from tests.sandbox.proxy.flaky import build_flaky

COMPANY = build_company(0)
SESSION = SessionInfo(session_id="sess_0001", agent_id="agent_support", principal_id="user_ops")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(slots=True)
class MemorySink:
    calls: list[RawCall] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    verdicts: list[tuple[str, DetectorResult]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def write_call(self, raw: RawCall) -> None:
        self.calls.append(raw)
        self.events.append(f"call:{raw.envelope_id}")

    def write_labels(self, labels: Sequence[Label]) -> None:
        self.labels.extend(labels)

    def write_verdict(self, envelope_id: str, result: DetectorResult) -> None:
        self.verdicts.append((envelope_id, result))
        self.events.append(f"verdict:{envelope_id}")


class SpyDownstream:
    """Delegates to a real client and records every forwarded call."""

    def __init__(self, inner: Client) -> None:
        self.inner = inner
        self.forwarded: list[tuple[str, dict[str, Any] | None]] = []

    async def list_tools(self) -> ListToolsResult:
        return await self.inner.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        self.forwarded.append((name, arguments))
        return await self.inner.call_tool(name, arguments)


class TickClock:
    def __init__(self, start: int = 1_700_000_000_000, step: int = 1000) -> None:
        self.now = start - step
        self.step = step

    def __call__(self) -> int:
        self.now += self.step
        return self.now


@asynccontextmanager
async def make_proxy(
    *,
    mutator: ActionMutator | None = None,
    shadow: Detector | None = None,
    timeout_s: float = 30.0,
    sink: Any = None,
) -> AsyncIterator[tuple[FirewallProxy, Any, Mapping[str, SpyDownstream]]]:
    """A proxy over fake zendesk + stripe + flaky servers, each behind a spying client."""
    servers = {
        "zendesk": zendesk_server.build_server(zendesk_server.FakeZendeskBackend(COMPANY)),
        "stripe": stripe_server.build_server(stripe_server.FakeStripeBackend(COMPANY)),
        "flaky": build_flaky(),
    }
    sink = MemorySink() if sink is None else sink
    async with AsyncExitStack() as stack:
        spies = {
            name: SpyDownstream(await stack.enter_async_context(Client(srv)))
            for name, srv in servers.items()
        }
        proxy = FirewallProxy(
            spies,
            SESSION,
            sink,
            mutator=mutator,
            shadow=shadow,
            timeout_s=timeout_s,
            clock_ms=TickClock(),
        )
        yield proxy, sink, spies
