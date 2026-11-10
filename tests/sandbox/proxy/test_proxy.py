"""Pass-through MCP proxy: namespacing, envelope logging, failures, D-014, shadow scoring."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson
import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from agentfw.core.engine import Engine
from agentfw.core.envelope import (
    ActionEnvelope,
    Decision,
    DetectorResult,
    Label,
    Outcome,
    RawCall,
    ResponseMeta,
)
from agentfw.core.patterns import is_id_like
from agentfw.sandbox.proxy_py import FirewallProxy, JsonlSink, MutationResult
from agentfw.sandbox.tools.stripe_server import fake_stripe_ids
from tests.sandbox.proxy.conftest import COMPANY, SESSION, MemorySink, make_proxy
from tests.sandbox.proxy.flaky import NOT_JSON

pytestmark = pytest.mark.anyio

TICKET = COMPANY.tickets[0].key
CHARGE_ID = fake_stripe_ids(COMPANY)[COMPANY.charges[0].key]
SYNTH = CallToolResult(
    content=[TextContent(type="text", text='{"id": "re_fake0000001", "amount": 1234.5}')],
    is_error=False,
)


def _text(res: CallToolResult) -> str:
    block = res.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def test_lists_namespaced_tools_and_forwards() -> None:
    async with make_proxy() as (proxy, sink, spies):
        names = {t.name for t in await proxy.list_tools()}
        assert {"stripe.retrieve_charge", "zendesk.get_ticket", "flaky.boom"} <= names
        assert all(n.count(".") == 1 for n in names)

        res = await proxy.call_tool("zendesk.get_ticket", {"ticket_id": TICKET})
        assert not res.is_error
        assert orjson.loads(_text(res))["id"] == TICKET
        assert spies["zendesk"].forwarded == [("get_ticket", {"ticket_id": TICKET})]

        raw = sink.calls[-1]
        assert raw.tool_id == "zendesk.get_ticket" and raw.seq_no == 0
        assert (raw.session_id, raw.agent_id, raw.principal_id) == (
            SESSION.session_id,
            SESSION.agent_id,
            SESSION.principal_id,
        )
        assert raw.args == {"ticket_id": TICKET}
        assert raw.ts_unix_ms == 1_700_000_000_000
        assert raw.response_meta is not None
        assert raw.response_meta.item_count == 1 and raw.response_meta.ids
        assert TICKET in raw.response_meta.ids
        assert all(is_id_like(i) for i in raw.response_meta.ids)
        assert raw.response_meta.error is False
        assert raw.response_meta.bytes == len(_text(res).encode())

        await proxy.call_tool("stripe.retrieve_charge", {"charge_id": CHARGE_ID})
        assert [c.seq_no for c in sink.calls] == [0, 1]
        assert sink.calls[1].ts_unix_ms > sink.calls[0].ts_unix_ms
        assert len({c.envelope_id for c in sink.calls}) == 2
        assert sink.labels == [] and sink.verdicts == []


async def test_downstream_failure_recorded_and_survivable() -> None:
    async with make_proxy(timeout_s=0.2) as (proxy, sink, _):
        res = await proxy.call_tool("flaky.boom", {})
        assert res.is_error and sink.calls[-1].response_meta.error is True
        assert not (await proxy.call_tool("flaky.ok", {})).is_error
        assert sink.calls[-1].response_meta.item_count == 2

        res = await proxy.call_tool("flaky.hang", {})
        assert res.is_error and "timed out" in _text(res)
        hung = sink.calls[-1]
        assert hung.tool_id == "flaky.hang" and hung.response_meta.error is True
        assert not (await proxy.call_tool("flaky.ok", {})).is_error

        res = await proxy.call_tool("flaky.text", {})
        assert not res.is_error
        meta = sink.calls[-1].response_meta
        assert meta.item_count is None and meta.bytes == len(NOT_JSON.encode())
        assert [c.seq_no for c in sink.calls] == [0, 1, 2, 3, 4]


async def test_downstream_exception_in_client_is_recorded() -> None:
    class Broken:
        async def list_tools(self) -> Any:
            from mcp.types import ListToolsResult, Tool

            return ListToolsResult(tools=[Tool(name="op", input_schema={"type": "object"})])

        async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
            raise ConnectionError("transport closed")

    sink = MemorySink()
    proxy = FirewallProxy({"broken": Broken()}, SESSION, sink)
    res = await proxy.call_tool("broken.op", {"x": 1})
    assert res.is_error and "ConnectionError" in _text(res)
    meta = sink.calls[-1].response_meta
    assert meta is not None and meta.error is True
    res = await proxy.call_tool("broken.op", {"x": 2})
    assert res.is_error and [c.seq_no for c in sink.calls] == [0, 1]


async def test_unknown_tool_is_error_without_envelope() -> None:
    async with make_proxy() as (proxy, sink, spies):
        for name in ("nope.get_ticket", "zendesk.nope", "get_ticket", "zendesk.", ".get_ticket"):
            res = await proxy.call_tool(name, {})
            assert res.is_error, name
        assert sink.calls == []
        assert all(s.forwarded == [] for s in spies.values())
        await proxy.call_tool("flaky.ok", {})
        assert sink.calls[-1].seq_no == 0


def test_server_names_must_be_lowercase_snake() -> None:
    for bad in ("Zendesk", "zen-desk", "zen.desk", ""):
        with pytest.raises(ValueError, match="server name"):
            FirewallProxy({bad: object()}, SESSION, MemorySink())  # type: ignore[dict-item]


async def test_build_server_round_trip() -> None:
    async with make_proxy() as (proxy, sink, _):
        async with Client(proxy.build_server()) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert "zendesk.get_ticket" in names
            res = await client.call_tool("zendesk.get_ticket", {"ticket_id": TICKET})
            assert not res.is_error and orjson.loads(_text(res))["id"] == TICKET
            assert (await client.call_tool("zendesk.missing", {})).is_error
        assert [c.tool_id for c in sink.calls] == ["zendesk.get_ticket"]


# --- mutation (D-014) -------------------------------------------------------------------------


class RefundMutator:
    """Scales stripe.create_refund amounts; everything else passes through untouched."""

    def __init__(
        self,
        *,
        forward: bool = False,
        synthetic: CallToolResult | None = SYNTH,
        change_args: bool = True,
        n_extra: int = 0,
        meta: ResponseMeta | None = None,
        target: str = "stripe.create_refund",
    ) -> None:
        self.forward = forward
        self.synthetic = synthetic
        self.change_args = change_args
        self.n_extra = n_extra
        self.meta = meta
        self.target = target
        self.seen: list[tuple[RawCall, tuple[RawCall, ...]]] = []

    def mutate(self, raw: RawCall, history: Sequence[RawCall]) -> MutationResult | None:
        self.seen.append((raw, tuple(history)))
        if raw.tool_id != self.target:
            return None
        args = dict(raw.args)
        if self.change_args:
            args["amount"] = 50 * float(args["amount"])  # type: ignore[arg-type]
        new = raw.model_copy(update={"args": args, "response_meta": self.meta})
        extras = [
            raw.model_copy(update={"envelope_id": f"{raw.envelope_id}x{i}", "seq_no": 0})
            for i in range(self.n_extra)
        ]
        label = Label(
            envelope_id=raw.envelope_id,
            label="attack",
            attack_family="magnitude_spike",
            injection_level="action",
            attack_id="atk_0001",
        )
        return MutationResult(
            raw=new,
            forward=self.forward,
            extra_calls=extras,
            labels=[label],
            synthetic_result=self.synthetic,
        )


async def test_unforwarded_mutation_returns_synthetic() -> None:
    mut = RefundMutator(n_extra=2)
    async with make_proxy(mutator=mut) as (proxy, sink, spies):
        await proxy.call_tool("zendesk.get_ticket", {"ticket_id": TICKET})
        res = await proxy.call_tool(
            "stripe.create_refund",
            {"charge_id": CHARGE_ID, "amount": 10.0, "reason": "requested_by_customer"},
        )
        assert res is SYNTH
        assert spies["stripe"].forwarded == []  # D-014: never reaches the backend

        first, refund, *extras = sink.calls
        assert refund.tool_id == "stripe.create_refund" and refund.args["amount"] == 500.0
        assert refund.response_meta is not None
        assert refund.response_meta.ids == ("re_fake0000001",)
        assert [c.seq_no for c in sink.calls] == [0, 1, 2, 3]
        assert [e.envelope_id for e in extras] == [
            f"{refund.envelope_id}x0",
            f"{refund.envelope_id}x1",
        ]
        assert [lab.envelope_id for lab in sink.labels] == [refund.envelope_id]

        # the mutator sees each call's final, response-carrying predecessors
        assert mut.seen[0][1] == ()
        assert mut.seen[1][1] == (first,)
        assert first.response_meta is not None and first.response_meta.item_count == 1


async def test_unforwarded_mutation_without_synthetic_is_error() -> None:
    async with make_proxy(mutator=RefundMutator(synthetic=None)) as (proxy, sink, spies):
        res = await proxy.call_tool(
            "stripe.create_refund", {"charge_id": CHARGE_ID, "amount": 1.0, "reason": "x"}
        )
        assert res.is_error
        assert spies["stripe"].forwarded == []
        assert sink.calls[-1].response_meta.error is True


async def test_mutated_args_are_never_forwarded_even_if_mutator_says_so() -> None:
    async with make_proxy(mutator=RefundMutator(forward=True)) as (proxy, sink, spies):
        res = await proxy.call_tool(
            "stripe.create_refund", {"charge_id": CHARGE_ID, "amount": 1.0, "reason": "x"}
        )
        assert res is SYNTH
        assert spies["stripe"].forwarded == []
        assert sink.calls[-1].args["amount"] == 50.0


async def test_forwarded_unmutated_call_keeps_mutator_response_meta() -> None:
    meta = ResponseMeta(item_count=500, bytes=10, error=False)
    mut = RefundMutator(
        forward=True, change_args=False, synthetic=None, meta=meta, target="flaky.ok"
    )
    async with make_proxy(mutator=mut) as (proxy, sink, spies):
        res = await proxy.call_tool("flaky.ok", {})
        assert not res.is_error and spies["flaky"].forwarded == [("ok", {})]
        assert sink.calls[-1].response_meta == meta
        assert len(sink.labels) == 1


# --- shadow scoring ---------------------------------------------------------------------------


class SpyDetector:
    def __init__(self, spies: dict[str, Any], sink: MemorySink, *, fail: bool = False) -> None:
        self.spies = spies
        self.sink = sink
        self.fail = fail
        self.log: list[tuple[Any, ...]] = []

    def _forwarded(self) -> int:
        return sum(len(s.forwarded) for s in self.spies.values())

    def reset(self) -> None:
        self.log.clear()

    def score(self, e: ActionEnvelope) -> DetectorResult:
        self.log.append(("score", e, self._forwarded(), len(self.sink.calls)))
        if self.fail:
            raise RuntimeError("detector crashed")
        return DetectorResult(
            fused_score=0.99,
            per_detector=(),
            reasons=(),
            latency_us=1.0,
            threshold=0.5,
            decision=Decision.BLOCK,
        )

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        self.log.append(("update", e, outcome, self._forwarded(), len(self.sink.calls)))
        if self.fail:
            raise RuntimeError("detector crashed")

    def state_bytes(self) -> int:
        return 0


async def test_shadow_scores_before_update() -> None:
    sink = MemorySink()
    spy: SpyDetector | None = None
    async with make_proxy(sink=sink) as (proxy, _, spies):
        spy = SpyDetector(dict(spies), sink)
        proxy.shadow = spy
        res = await proxy.call_tool("zendesk.get_ticket", {"ticket_id": TICKET})
        assert not res.is_error  # shadow never blocks, even on a BLOCK decision
        await proxy.call_tool("flaky.ok", {})

    kinds = [entry[0] for entry in spy.log]
    assert kinds == ["score", "update", "score", "update"]
    (_, s0, fwd_s0, calls_s0), (_, u0, outcome, fwd_u0, _) = spy.log[0], spy.log[1]
    assert fwd_s0 == 0 and calls_s0 == 0  # scored before forwarding and before logging
    assert s0.response_meta is None and s0.prior_tools == ()
    assert fwd_u0 == 1 and outcome is Outcome.ALLOWED
    assert u0.envelope_id == s0.envelope_id
    assert u0.response_meta is not None and u0.response_meta.item_count == 1

    s1 = spy.log[2][1]
    assert s1.prior_tools == ("zendesk.get_ticket",)
    assert s1.prev_response is not None and s1.prev_response.item_count == 1
    assert TICKET in s1.session_ids_seen  # ids from the previous response are observed

    assert [eid for eid, _ in sink.verdicts] == [c.envelope_id for c in sink.calls]
    assert sink.events.index(f"verdict:{s0.envelope_id}") < sink.events.index(
        f"call:{s0.envelope_id}"
    )


async def test_shadow_failure_never_blocks() -> None:
    sink = MemorySink()
    async with make_proxy(sink=sink) as (proxy, _, spies):
        proxy.shadow = SpyDetector(dict(spies), sink, fail=True)
        res = await proxy.call_tool("zendesk.get_ticket", {"ticket_id": TICKET})
        assert not res.is_error
        assert spies["zendesk"].forwarded and len(sink.calls) == 1
        assert sink.verdicts == []


async def test_shadow_engine_end_to_end_writes_jsonl(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path)
    mut = RefundMutator(n_extra=1)
    async with make_proxy(sink=sink, shadow=Engine(), mutator=mut) as (proxy, _, _spies):
        await proxy.call_tool("zendesk.get_ticket", {"ticket_id": TICKET})
        await proxy.call_tool("stripe.retrieve_charge", {"charge_id": CHARGE_ID})
        await proxy.call_tool(
            "stripe.create_refund", {"charge_id": CHARGE_ID, "amount": 2.0, "reason": "x"}
        )
        await proxy.call_tool("flaky.boom", {})

    calls = [
        RawCall.model_validate_json(line)
        for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    verdicts = [
        orjson.loads(line) for line in (tmp_path / "verdicts.jsonl").read_text().splitlines()
    ]
    labels = [
        Label.model_validate_json(line)
        for line in (tmp_path / "labels.jsonl").read_text().splitlines()
    ]
    assert [c.seq_no for c in calls] == [0, 1, 2, 3, 4]
    assert [v["envelope_id"] for v in verdicts] == [c.envelope_id for c in calls]
    assert all(v["mode"] == "shadow" for v in verdicts)
    assert all(0.0 <= v["result"]["fused_score"] <= 1.0 for v in verdicts)
    assert [lab.envelope_id for lab in labels] == [calls[2].envelope_id]
    assert calls[-1].response_meta is not None and calls[-1].response_meta.error is True
