"""LLM client seam for the sandbox agent: an OpenAI-compatible client, a scripted double, metering.

The agent model is an open-source model (Qwen 2.5 72B, Llama 3.3 70B, ...) behind an
OpenAI-compatible endpoint, so tool calling is the provider's native ``tools=`` mechanism.

Provider function names must match ``^[a-zA-Z0-9_-]{1,64}$`` (global constraint R5), which
excludes the ``.`` of ``server.tool`` ids; ``to_fn_name`` / ``from_fn_name`` map ``.`` <-> ``__``.
The mapping round-trips as long as no server or tool name contains ``__`` (a double underscore),
which holds for every catalog id and every sandbox server tool (both are tested).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import openai

__all__ = [
    "ChatClient",
    "ChatTurn",
    "OpenAICompatClient",
    "ScriptedChatClient",
    "ToolCall",
    "Usage",
    "UsageMeter",
    "from_fn_name",
    "to_fn_name",
]

_FN_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


def to_fn_name(tool_id: str) -> str:
    """``stripe.create_refund`` -> ``stripe__create_refund`` (a legal provider function name)."""
    name = tool_id.replace(".", "__")
    if not _FN_NAME.fullmatch(name):
        raise ValueError(f"tool id {tool_id!r} has no legal provider function name ({name!r})")
    return name


def from_fn_name(name: str) -> str:
    return name.replace("__", ".")


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class ChatTurn:
    content: str | None
    tool_calls: list[ToolCall]
    usage: Usage


class ChatClient(Protocol):
    async def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn: ...


class OpenAICompatClient:
    """Chat completions with native tool calling against any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
    ) -> None:
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=cast("Any", messages),
            temperature=self._temperature,
            **kwargs,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments)
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        ]
        u = resp.usage
        usage = Usage(u.prompt_tokens, u.completion_tokens) if u is not None else Usage(0, 0)
        return ChatTurn(content=msg.content, tool_calls=calls, usage=usage)


@dataclass(slots=True)
class ScriptedChatClient:
    """Test double: replays ``turns`` in order and records each call's (messages, tools)."""

    turns: list[ChatTurn]
    calls: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = field(
        default_factory=list
    )

    async def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        self.calls.append((list(messages), list(tools)))
        if not self.turns:
            raise RuntimeError("ScriptedChatClient exhausted: no scripted turn left")
        return self.turns.pop(0)


class UsageMeter:
    """Accumulates token usage and its USD cost at per-million-token prices."""

    def __init__(self, price_in_per_m: float, price_out_per_m: float) -> None:
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self.usd = 0.0

    def add(self, u: Usage) -> None:
        self.usd += (
            u.prompt_tokens * self.price_in_per_m / 1e6
            + u.completion_tokens * self.price_out_per_m / 1e6
        )
