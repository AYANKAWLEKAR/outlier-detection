"""Pass-through MCP proxy (spec 1.2, 2.2, 2.3): every tool call becomes a benchmark envelope.

The proxy fronts several downstream MCP servers and re-exposes their tools as ``server.tool``.
For each call it

1. builds a ``RawCall`` (``seq_no`` from a per-proxy counter, ``ts_unix_ms`` from ``clock_ms``);
2. lets the optional ``ActionMutator`` act. **D-014:** a call whose ``tool_id`` or ``args`` the
   mutator changed is never forwarded, whatever ``forward`` says; ``forward=False`` returns the
   mutator's ``synthetic_result`` (or an error result if it gave none). The downstream is not
   called in either case;
3. with a ``shadow`` detector, normalizes the call (no response yet) and ``score``\\s it *before*
   forwarding, then ``write_verdict``\\s the result;
4. forwards it with a timeout. A downstream that raises, times out or returns non-JSON yields an
   ``is_error`` result (``isError`` on the wire) and an envelope with ``response_meta.error``; the
   proxy keeps serving the session;
5. extracts ``response_meta``, writes the final call (and the mutator's labels), and only then
   ``update``\\s the shadow detector with the response-carrying envelope (same ``envelope_id``)
   and ``Outcome.ALLOWED``, and advances the session tracker (score before update, §2.1).

**Shadow mode never blocks.** The detector's decision is recorded and otherwise ignored, every
call is forwarded and learned as ``ALLOWED`` (§2.2), and a detector that raises is logged and
skipped. Unknown tools return an error result and produce no envelope.

Calls are serialized per proxy (one agent session), so ``seq_no``, the mutator's history and the
session tracker stay causal. Once admitted, a call runs shielded from upstream cancellation (an MCP
client that gives up sends ``notifications/cancelled``): the forward is still bounded by
``timeout_s`` and the call is always recorded, so there is no lost envelope or ``seq_no`` gap.
Mutator output is re-validated as ``RawCall``; invalid output is rejected (fail closed: the
original call is recorded with an error result and nothing is forwarded or labelled).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio
import numpy as np
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)
from pydantic import ValidationError

from agentfw.bench.protocol import Detector
from agentfw.core.envelope import Label, Outcome, RawCall
from agentfw.core.ids import new_ulid
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.sandbox.proxy_py.response_meta import extract_response_meta
from agentfw.sandbox.proxy_py.sinks import EnvelopeSink

__all__ = [
    "NAME_PART",
    "ActionMutator",
    "Downstream",
    "FirewallProxy",
    "MutationResult",
    "SessionInfo",
    "error_result",
    "wall_ms",
]

logger = logging.getLogger(__name__)

NAME_PART = re.compile(r"[a-z0-9_]+")
"""A server or tool name part; ``server.tool`` must match ``RawCall.tool_id``."""


def wall_ms() -> int:
    return time.time_ns() // 1_000_000


class Downstream(Protocol):
    """What the proxy needs from a downstream connection (``mcp.Client`` or ``ClientSession``)."""

    async def list_tools(self) -> ListToolsResult: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult: ...


@dataclass(slots=True, frozen=True)
class SessionInfo:
    session_id: str
    agent_id: str
    principal_id: str


@dataclass(slots=True)
class MutationResult:
    raw: RawCall
    forward: bool
    extra_calls: list[RawCall] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    synthetic_result: CallToolResult | None = None


class ActionMutator(Protocol):
    def mutate(self, raw: RawCall, history: Sequence[RawCall]) -> MutationResult | None:
        """Return ``None`` to pass ``raw`` through untouched.

        ``history`` is this session's earlier final calls (with ``response_meta``). If the
        returned ``raw`` carries a ``response_meta``, it replaces the extracted one.
        ``extra_calls`` are logged after the call, never forwarded, and get fresh ``seq_no``\\s.
        """
        ...


def _invalid_mutation(m: MutationResult) -> str | None:
    """Re-validate mutator output (``model_copy`` skips validation); the error text, or None."""
    try:
        for r in (m.raw, *m.extra_calls):
            RawCall.model_validate(r.model_dump())
    except ValidationError as exc:
        return str(exc)
    return None


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


class FirewallProxy:
    def __init__(
        self,
        downstreams: Mapping[str, Downstream],
        session: SessionInfo,
        sink: EnvelopeSink,
        *,
        mutator: ActionMutator | None = None,
        shadow: Detector | None = None,
        timeout_s: float = 30.0,
        clock_ms: Callable[[], int] = wall_ms,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name in downstreams:
            if not NAME_PART.fullmatch(name):
                raise ValueError(
                    f"server name {name!r} must be lowercase snake ([a-z0-9_]+) to form tool ids"
                )
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self.downstreams = dict(downstreams)
        self.session = session
        self.sink = sink
        self.mutator = mutator
        self.shadow = shadow
        self.timeout_s = timeout_s
        self.clock_ms = clock_ms
        self._rng = np.random.default_rng() if rng is None else rng
        self._routes: dict[str, tuple[str, str]] | None = None
        self._seq = 0
        self._history: list[RawCall] = []
        self._tracker = SessionTracker()
        self._lock = anyio.Lock()

    # -- discovery ---------------------------------------------------------------------------

    async def list_tools(self) -> list[Tool]:
        """Every downstream tool, renamed ``server.tool``; refreshes the routing table."""
        tools: list[Tool] = []
        routes: dict[str, tuple[str, str]] = {}
        for server, ds in self.downstreams.items():
            for t in (await ds.list_tools()).tools:
                if not NAME_PART.fullmatch(t.name):
                    logger.warning("skipping %s tool %r: not a valid tool id part", server, t.name)
                    continue
                name = f"{server}.{t.name}"
                routes[name] = (server, t.name)
                tools.append(t.model_copy(update={"name": name}))
        self._routes = routes
        return tools

    # -- the call path -----------------------------------------------------------------------

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> CallToolResult:
        if self._routes is None:
            await self.list_tools()
        assert self._routes is not None
        route = self._routes.get(name)
        if route is None:
            return error_result(f"unknown tool {name!r}")
        async with self._lock:
            # Shielded: once a call is admitted it is always recorded, even if the upstream client
            # cancels (notifications/cancelled) mid-forward; fail_after(timeout_s) bounds it.
            with anyio.CancelScope(shield=True):
                result = await self._call(name, route, dict(arguments or {}))
        return result

    async def _call(
        self, name: str, route: tuple[str, str], arguments: dict[str, Any]
    ) -> CallToolResult:
        raw = self._new_raw(name, arguments)
        mutation = (
            self.mutator.mutate(raw, tuple(self._history)) if self.mutator is not None else None
        )
        if mutation is not None and (invalid := _invalid_mutation(mutation)) is not None:
            logger.error("rejecting mutator output for %s: %s", raw.envelope_id, invalid)
            self._shadow_score(raw)
            result = error_result(f"{name}: mutator produced an invalid call; not forwarded")
            self._record(raw.model_copy(update={"response_meta": extract_response_meta(result)}))
            return result
        forward = True
        if mutation is not None:
            mutated = mutation.raw.tool_id != raw.tool_id or mutation.raw.args != raw.args
            if mutation.forward and mutated:
                logger.warning(
                    "D-014: %s was mutated; not forwarding despite forward=True", raw.envelope_id
                )
            forward = mutation.forward and not mutated  # D-014
            raw = mutation.raw.model_copy(update={"response_meta": None})

        self._shadow_score(raw)
        if forward:
            result = await self._forward(*route, arguments)
        elif mutation is not None and mutation.synthetic_result is not None:
            result = mutation.synthetic_result
        else:
            result = error_result(f"{name}: injected call not forwarded (D-014)")

        override = mutation.raw.response_meta if mutation is not None else None
        meta = override if override is not None else extract_response_meta(result)
        self._record(raw.model_copy(update={"response_meta": meta}))
        if mutation is not None:
            self.sink.write_labels(mutation.labels)
            for extra in mutation.extra_calls:
                extra = extra.model_copy(update={"seq_no": self._next_seq()})
                self._shadow_score(extra.model_copy(update={"response_meta": None}))
                self._record(extra)
        return result

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _new_raw(self, tool_id: str, args: dict[str, Any]) -> RawCall:
        ts = self.clock_ms()
        return RawCall(
            envelope_id=new_ulid(ts, self._rng),
            session_id=self.session.session_id,
            agent_id=self.session.agent_id,
            principal_id=self.session.principal_id,
            seq_no=self._next_seq(),
            ts_unix_ms=ts,
            tool_id=tool_id,
            args=args,
        )

    async def _forward(self, server: str, tool: str, args: dict[str, Any]) -> CallToolResult:
        try:
            with anyio.fail_after(self.timeout_s):
                return await self.downstreams[server].call_tool(tool, args)
        except TimeoutError:
            logger.warning("%s.%s timed out after %g s", server, tool, self.timeout_s)
            return error_result(f"{server}.{tool} timed out after {self.timeout_s:g} s")
        except Exception as exc:  # boundary: a failing downstream must not end the session
            logger.warning("%s.%s failed: %r", server, tool, exc)
            return error_result(f"{server}.{tool} failed: {type(exc).__name__}: {exc}")

    def _shadow_score(self, raw: RawCall) -> None:
        """Score ``raw`` (no response yet) against state as of before it. Never blocks."""
        if self.shadow is None:
            return
        env = normalize(raw, self._tracker)
        try:
            result = self.shadow.score(env)
        except Exception:  # boundary: shadow mode must never affect the call
            logger.exception("shadow score failed for %s", raw.envelope_id)
            return
        self.sink.write_verdict(raw.envelope_id, result)

    def _record(self, final: RawCall) -> None:
        """Log the response-carrying call, then update the shadow detector and the tracker."""
        self.sink.write_call(final)
        self._history.append(final)
        env = normalize(final, self._tracker)
        if self.shadow is not None:
            try:
                self.shadow.update(env, Outcome.ALLOWED)
            except Exception:  # boundary: shadow mode must never affect the call
                logger.exception("shadow update failed for %s", final.envelope_id)
        self._tracker.observe(env)

    # -- MCP server --------------------------------------------------------------------------

    def build_server(self, name: str = "agentfw-proxy") -> Server[Any]:
        """A low-level MCP server whose tool list is discovered from the downstreams at runtime."""

        async def on_list_tools(
            ctx: ServerRequestContext[Any], params: PaginatedRequestParams | None
        ) -> ListToolsResult:
            return ListToolsResult(tools=await self.list_tools())

        async def on_call_tool(
            ctx: ServerRequestContext[Any], params: CallToolRequestParams
        ) -> CallToolResult:
            return await self.call_tool(params.name, params.arguments)

        return Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)
