"""A thin, deliberately boring ReAct loop (spec 1.2) over native tool calling, tools via MCP.

Each iteration is one LLM completion (a *step*). A turn with no tool calls ends the session
(``final``); otherwise every tool call is executed through MCP and answered with a ``role="tool"``
message carrying its ``tool_call_id``. Failures go back to the model, they never crash the loop:

* error results (``is_error``) are passed to the model unchanged (AG1);
* malformed (or non-object) JSON arguments -> ``invalid JSON arguments: ...``, tool not called;
* an unknown function name -> ``ERROR: unknown tool ...``; the tool is not called;
* an exception raised by the MCP call -> ``ERROR: <type>: <message>``.

An exception from the LLM ends the session with ``stopped="error"`` (and ``Transcript.error``);
the runner records such a session as ``failed``. After ``max_steps`` steps
without a final answer the session ends with ``stopped="max_steps"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import orjson
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from agentfw.sandbox.llm import ChatClient, Usage, UsageMeter, to_fn_name

__all__ = ["McpLike", "Transcript", "run_session"]

logger = logging.getLogger(__name__)


class McpLike(Protocol):
    """The minimal MCP surface the loop needs (``mcp.Client``, ``FirewallProxy``, test doubles).

    ``list_tools`` is read once per session; names must be ``server.tool`` ids or any name that
    ``to_fn_name`` can make legal.
    """

    async def list_tools(self) -> ListToolsResult: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult: ...


@dataclass(slots=True)
class Transcript:
    messages: list[dict[str, object]] = field(default_factory=list)
    steps: int = 0
    stopped: Literal["final", "max_steps", "error"] = "final"
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    error: str | None = None  # ``stopped="error"``: the LLM failure, ``Type: message``


def _result_text(res: CallToolResult) -> str:
    text = "\n".join(b.text for b in res.content if isinstance(b, TextContent))
    if not text and not res.is_error and res.structured_content:
        return orjson.dumps(res.structured_content, option=orjson.OPT_SORT_KEYS).decode()
    return text or "(no text output)"


async def _execute(mcp: McpLike, tool_ids: dict[str, str], name: str, arguments_json: str) -> str:
    tool_id = tool_ids.get(name)
    if tool_id is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        args = orjson.loads(arguments_json) if (arguments_json or "").strip() else {}
    except orjson.JSONDecodeError as exc:
        return f"invalid JSON arguments: {exc}"
    if not isinstance(args, dict):
        return f"invalid JSON arguments: expected a JSON object, got {type(args).__name__}"
    try:
        return _result_text(await mcp.call_tool(tool_id, args))
    except Exception as exc:  # boundary: a failing tool is reported to the model, not raised
        logger.warning("tool %s raised %r", tool_id, exc)
        return f"ERROR: {type(exc).__name__}: {exc}"


async def run_session(
    llm: ChatClient,
    mcp: McpLike,
    system_prompt: str,
    task_prompt: str,
    *,
    max_steps: int = 25,
    meter: UsageMeter | None = None,
) -> Transcript:
    candidates: dict[str, list[Tool]] = {}
    for t in (await mcp.list_tools()).tools:
        try:
            candidates.setdefault(to_fn_name(t.name), []).append(t)
        except ValueError as exc:
            logger.warning("skipping tool %r: %s", t.name, exc)
    tool_ids: dict[str, str] = {}
    tools: list[dict[str, object]] = []
    for fn, group in candidates.items():
        if len(group) > 1:  # neither may be callable: the model could invoke the wrong one
            logger.warning("skipping tools %s: same function name %r", [t.name for t in group], fn)
            continue
        t = group[0]
        tool_ids[fn] = t.name
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
        )
    tr = Transcript(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
    )
    tr.stopped = "max_steps"
    while tr.steps < max_steps:
        try:
            turn = await llm.complete(tr.messages, tools)
        except Exception as exc:  # boundary: provider failure ends the session, caller decides
            logger.exception("LLM call failed at step %d", tr.steps)
            tr.stopped = "error"
            tr.error = f"LLM call failed at step {tr.steps}: {type(exc).__name__}: {exc}"
            return tr
        tr.steps += 1
        tr.usage.prompt_tokens += turn.usage.prompt_tokens
        tr.usage.completion_tokens += turn.usage.completion_tokens
        if meter is not None:
            meter.add(turn.usage)
        assistant: dict[str, object] = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json},
                }
                for tc in turn.tool_calls
            ]
        tr.messages.append(assistant)
        if not turn.tool_calls:
            tr.stopped = "final"
            return tr
        for tc in turn.tool_calls:
            content = await _execute(mcp, tool_ids, tc.name, tc.arguments_json)
            tr.messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return tr
