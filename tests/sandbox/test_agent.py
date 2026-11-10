"""ReAct loop: scripted sessions through the in-memory proxy, name mapping, failures, metering."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import orjson
import pytest
from mcp import Client
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from agentfw.sandbox.agent import run_session
from agentfw.sandbox.llm import (
    ChatTurn,
    ScriptedChatClient,
    ToolCall,
    Usage,
    UsageMeter,
    from_fn_name,
    to_fn_name,
)
from agentfw.sandbox.tools import (
    mail_server,
    salesforce_server,
    slack_server,
    stripe_server,
    zendesk_server,
)
from agentfw.synth.load import load_catalog
from tests.sandbox.proxy.conftest import COMPANY, MemorySink, make_proxy

pytestmark = pytest.mark.anyio

FN_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
TICKET = COMPANY.tickets[0].key
CHARGE_ID = stripe_server.fake_stripe_ids(COMPANY)[COMPANY.charges[0].key]


@asynccontextmanager
async def agent_session() -> AsyncIterator[tuple[Client, MemorySink, Any]]:
    """An mcp.Client connected to the firewall proxy's MCP server (the agent's real wiring)."""
    async with (
        make_proxy() as (proxy, sink, spies),
        Client(proxy.build_server()) as client,
    ):
        yield client, sink, spies


def _call(i: str, name: str, args: dict[str, Any] | str) -> ToolCall:
    raw = args if isinstance(args, str) else orjson.dumps(args).decode()
    return ToolCall(id=i, name=name, arguments_json=raw)


def _turn(*calls: ToolCall, content: str | None = None, p: int = 10, c: int = 5) -> ChatTurn:
    return ChatTurn(content=content, tool_calls=list(calls), usage=Usage(p, c))


def _tool_msgs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] == "tool"]


async def test_scripted_three_turn_session_yields_two_envelopes() -> None:
    llm = ScriptedChatClient(
        [
            _turn(_call("c1", "zendesk__get_ticket", {"ticket_id": TICKET})),
            _turn(
                _call(
                    "c2",
                    "stripe__create_refund",
                    {"charge_id": CHARGE_ID, "amount": 5.0, "reason": "requested_by_customer"},
                )
            ),
            _turn(content="Refund issued."),
        ]
    )
    async with agent_session() as (client, sink, _spies):
        t = await run_session(llm, client, "You are support.", "Handle the ticket.")
    assert t.stopped == "final"
    assert t.steps == 3
    assert [c.tool_id for c in sink.calls] == ["zendesk.get_ticket", "stripe.create_refund"]
    assert t.usage == Usage(30, 15)
    # the model saw tool definitions with provider-legal names, on every call
    tools = llm.calls[0][1]
    names = {str(t_["function"]["name"]) for t_ in tools}  # type: ignore[index]
    assert {"zendesk__get_ticket", "stripe__create_refund"} <= names
    # tool results fed back with tool_call_id
    tm = _tool_msgs(t.messages)
    assert [m["tool_call_id"] for m in tm] == ["c1", "c2"]
    assert TICKET in tm[0]["content"]
    assert t.messages[0] == {"role": "system", "content": "You are support."}
    assert t.messages[1] == {"role": "user", "content": "Handle the ticket."}
    assert t.messages[-1]["role"] == "assistant" and t.messages[-1]["content"] == "Refund issued."
    # assistant message carrying tool_calls uses the provider function name
    asst = [m for m in t.messages if m["role"] == "assistant" and m.get("tool_calls")]
    first_call: Any = asst[0]["tool_calls"]
    assert first_call[0]["function"]["name"] == "zendesk__get_ticket"


async def test_error_result_returned_verbatim() -> None:
    llm = ScriptedChatClient(
        [_turn(_call("c1", "flaky__boom", {})), _turn(content="gave up")],
    )
    async with agent_session() as (client, _sink, _spies):
        t = await run_session(llm, client, "s", "t")
    (msg,) = _tool_msgs(t.messages)
    assert not msg["content"].startswith("ERROR")  # AG1: error text passes through unchanged
    assert msg["content"] == "Error executing tool boom"
    assert t.stopped == "final"


async def test_malformed_json_goes_back_without_calling_tool() -> None:
    llm = ScriptedChatClient(
        [_turn(_call("c1", "zendesk__get_ticket", "{not json")), _turn(content="ok")]
    )
    async with agent_session() as (client, sink, spies):
        t = await run_session(llm, client, "s", "t")
    (msg,) = _tool_msgs(t.messages)
    assert msg["content"].startswith("invalid JSON arguments: ")
    assert sink.calls == [] and spies["zendesk"].forwarded == []
    assert t.stopped == "final"


async def test_non_object_json_arguments_rejected() -> None:
    llm = ScriptedChatClient(
        [_turn(_call("c1", "zendesk__get_ticket", "[1, 2]")), _turn(content="ok")]
    )
    async with agent_session() as (client, sink, _spies):
        t = await run_session(llm, client, "s", "t")
    assert _tool_msgs(t.messages)[0]["content"].startswith("invalid JSON arguments: ")
    assert sink.calls == []


async def test_empty_arguments_string_means_no_args() -> None:
    llm = ScriptedChatClient([_turn(_call("c1", "flaky__boom", "")), _turn(content="ok")])
    async with agent_session() as (client, sink, _spies):
        await run_session(llm, client, "s", "t")
    assert [c.tool_id for c in sink.calls] == ["flaky.boom"]


async def test_unknown_function_name_errors_without_calling() -> None:
    llm = ScriptedChatClient([_turn(_call("c1", "nope__nothing", {})), _turn(content="ok")])
    async with agent_session() as (client, sink, _spies):
        t = await run_session(llm, client, "s", "t")
    (msg,) = _tool_msgs(t.messages)
    assert "unknown tool" in msg["content"] and "nope__nothing" in msg["content"]
    assert sink.calls == []


async def test_stops_at_max_steps() -> None:
    forever = [_turn(_call(f"c{i}", "flaky__boom", {})) for i in range(10)]
    llm = ScriptedChatClient(forever)
    async with agent_session() as (client, sink, _spies):
        t = await run_session(llm, client, "s", "t", max_steps=3)
    assert t.stopped == "max_steps"
    assert t.steps == 3
    assert len(llm.calls) == 3 and len(sink.calls) == 3
    assert t.messages[-1]["role"] == "tool"


async def test_multiple_tool_calls_in_one_turn_all_answered() -> None:
    llm = ScriptedChatClient(
        [
            _turn(
                _call("a", "zendesk__get_ticket", {"ticket_id": TICKET}),
                _call("b", "zendesk__get_ticket", {"ticket_id": TICKET}),
            ),
            _turn(content="done"),
        ]
    )
    async with agent_session() as (client, sink, _spies):
        t = await run_session(llm, client, "s", "t")
    assert [m["tool_call_id"] for m in _tool_msgs(t.messages)] == ["a", "b"]
    assert len(sink.calls) == 2


async def test_llm_exception_stops_with_error() -> None:
    class Boom:
        async def complete(self, messages: Any, tools: Any) -> ChatTurn:
            raise RuntimeError("provider down")

    async with agent_session() as (client, _sink, _spies):
        t = await run_session(Boom(), client, "s", "t")
    assert t.stopped == "error"


async def test_meter_accumulates_usd_and_usage() -> None:
    llm = ScriptedChatClient(
        [
            _turn(_call("c1", "flaky__boom", {}), p=1_000_000, c=0),
            _turn(content="x", p=0, c=500_000),
        ]
    )
    meter = UsageMeter(price_in_per_m=2.0, price_out_per_m=10.0)
    async with agent_session() as (client, _sink, _spies):
        t = await run_session(llm, client, "s", "t", meter=meter)
    assert meter.usd == pytest.approx(2.0 + 5.0)
    assert t.usage == Usage(1_000_000, 500_000)


def test_meter_add_formula() -> None:
    m = UsageMeter(0.5, 1.5)
    m.add(Usage(2_000, 1_000))
    m.add(Usage(0, 1_000))
    assert m.usd == pytest.approx(2_000 * 0.5 / 1e6 + 2_000 * 1.5 / 1e6)


def test_scripted_client_exhausted_raises() -> None:
    import anyio

    async def go() -> None:
        await ScriptedChatClient([]).complete([], [])

    with pytest.raises(RuntimeError, match="exhausted"):
        anyio.run(go)


def test_fn_name_mapping() -> None:
    assert to_fn_name("stripe.create_refund") == "stripe__create_refund"
    assert from_fn_name("stripe__create_refund") == "stripe.create_refund"


def test_catalog_tool_ids_map_to_legal_names_and_round_trip() -> None:
    ids = list(load_catalog())
    assert len(ids) > 10
    for tid in ids:
        name = to_fn_name(tid)
        assert FN_NAME.fullmatch(name), name
        assert from_fn_name(name) == tid


async def test_every_sandbox_server_tool_maps_legally() -> None:
    servers = {
        "zendesk": zendesk_server.build_server(zendesk_server.FakeZendeskBackend(COMPANY)),
        "stripe": stripe_server.build_server(stripe_server.FakeStripeBackend(COMPANY)),
        "salesforce": salesforce_server.build_server(
            salesforce_server.FakeSalesforceBackend(COMPANY)
        ),
        "slack": slack_server.build_server(slack_server.FakeSlackBackend("C0TEST")),
        "mail": mail_server.build_server(mail_server.FakeMailBackend()),
    }
    seen = 0
    for server, srv in servers.items():
        async with Client(srv) as c:
            for t in (await c.list_tools()).tools:
                tid = f"{server}.{t.name}"
                assert FN_NAME.fullmatch(to_fn_name(tid)), tid
                assert from_fn_name(to_fn_name(tid)) == tid
                seen += 1
    assert seen > 10


class _FakeMcp:
    """Minimal McpLike double with configurable tools and call behaviour."""

    def __init__(self, names: list[str], result: CallToolResult | Exception | None = None) -> None:
        self.tools = [
            Tool(name=n, description="d", input_schema={"type": "object", "properties": {}})
            for n in names
        ]
        self.result = result
        self.called: list[str] = []

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=self.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        self.called.append(name)
        if isinstance(self.result, Exception):
            raise self.result
        assert self.result is not None
        return self.result


def _offered(llm: ScriptedChatClient) -> set[str]:
    return {str(t["function"]["name"]) for t in llm.calls[0][1]}  # type: ignore[index]


async def test_illegal_tool_name_skipped_session_survives() -> None:
    mcp = _FakeMcp(["good.tool", "bad.name with space", "x." + "a" * 70])
    llm = ScriptedChatClient([_turn(content="done")])
    t = await run_session(llm, mcp, "s", "t")
    assert t.stopped == "final"
    assert _offered(llm) == {"good__tool"}


async def test_colliding_fn_names_both_skipped() -> None:
    # "a.b__c" and "a__b.c" both map to "a__b__c"
    mcp = _FakeMcp(["a.b__c", "a__b.c", "ok.tool"])
    llm = ScriptedChatClient([_turn(_call("c1", "a__b__c", {})), _turn(content="done")])
    t = await run_session(llm, mcp, "s", "t")
    assert _offered(llm) == {"ok__tool"}
    assert mcp.called == []
    assert "unknown tool" in _tool_msgs(t.messages)[0]["content"]


async def test_mcp_call_raising_is_reported_and_session_continues() -> None:
    mcp = _FakeMcp(["s.t"], RuntimeError("transport gone"))
    llm = ScriptedChatClient([_turn(_call("c1", "s__t", {})), _turn(content="done")])
    t = await run_session(llm, mcp, "s", "t")
    (msg,) = _tool_msgs(t.messages)
    assert msg["tool_call_id"] == "c1" and "transport gone" in msg["content"]
    assert t.stopped == "final"


async def test_none_arguments_treated_as_empty() -> None:
    mcp = _FakeMcp(["s.t"], CallToolResult(content=[TextContent(type="text", text="ok")]))
    tc = ToolCall(id="c1", name="s__t", arguments_json=None)  # type: ignore[arg-type]
    t = await run_session(ScriptedChatClient([_turn(tc), _turn(content="d")]), mcp, "s", "t")
    assert mcp.called == ["s.t"]
    assert _tool_msgs(t.messages)[0]["content"] == "ok"


async def test_empty_success_result_gets_placeholder_or_structured() -> None:
    mcp = _FakeMcp(["s.t"], CallToolResult(content=[]))
    t = await run_session(
        ScriptedChatClient([_turn(_call("c", "s__t", {})), _turn(content="d")]), mcp, "s", "t"
    )
    assert _tool_msgs(t.messages)[0]["content"] == "(no text output)"
    mcp = _FakeMcp(["s.t"], CallToolResult(content=[], structured_content={"k": 1}))
    t = await run_session(
        ScriptedChatClient([_turn(_call("c", "s__t", {})), _turn(content="d")]), mcp, "s", "t"
    )
    assert orjson.loads(_tool_msgs(t.messages)[0]["content"]) == {"k": 1}


def test_to_fn_name_rejects_illegal() -> None:
    with pytest.raises(ValueError):
        to_fn_name("a.b c")
    with pytest.raises(ValueError):
        to_fn_name("a." + "x" * 64)


async def test_final_turn_with_no_content() -> None:
    mcp = _FakeMcp([])
    t = await run_session(ScriptedChatClient([_turn(content=None)]), mcp, "s", "t")
    assert t.stopped == "final" and t.messages[-1]["content"] is None


async def test_max_steps_zero_makes_no_llm_call() -> None:
    llm = ScriptedChatClient([_turn(content="x")])
    t = await run_session(llm, _FakeMcp([]), "s", "t", max_steps=0)
    assert t.stopped == "max_steps" and t.steps == 0 and llm.calls == []


def test_scripted_client_does_not_mutate_callers_list() -> None:
    import anyio

    turns = [_turn(content="x")]
    client = ScriptedChatClient(turns)
    anyio.run(client.complete, [], [])
    assert len(turns) == 1
