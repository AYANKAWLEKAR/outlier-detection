"""OpenAICompatClient offline: a provider reply without ``usage`` must not meter as free."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import orjson
import pytest

from agentfw.sandbox.llm import OpenAICompatClient, Usage, UsageMeter

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Completions:
    def __init__(self, reply: Any) -> None:
        self.reply = reply

    async def create(self, **kwargs: Any) -> Any:
        return self.reply


def _client(content: str | None, usage: Any, tool_calls: Any = None) -> OpenAICompatClient:
    client = OpenAICompatClient("http://llm.test/v1", "test-key", "test-model")
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(reply)))  # type: ignore[assignment]
    return client


_MESSAGES: list[dict[str, object]] = [
    {"role": "system", "content": "You are a support agent. " * 20},
    {"role": "user", "content": "Refund the duplicate charge on ticket T-1."},
]


async def test_missing_usage_is_estimated_conservatively_not_zero() -> None:
    content = "Done: the refund was issued and the ticket is closed. " * 4
    turn = await _client(content, None).complete(_MESSAGES, [])
    sent = len(orjson.dumps(_MESSAGES))
    # at least the common ~4 chars/token rate, so the budget never under-counts the spend
    assert turn.usage.prompt_tokens >= sent / 4
    assert turn.usage.completion_tokens >= len(content) / 4
    meter = UsageMeter(1.0, 1.0)
    meter.add(turn.usage)
    assert meter.usd > 0


async def test_missing_usage_counts_tool_call_arguments() -> None:
    tc = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name="stripe__create_refund", arguments='{"amount": 12.5}' * 8),
    )
    turn = await _client(None, None, [tc]).complete(_MESSAGES, [])
    assert turn.usage.completion_tokens >= len('{"amount": 12.5}' * 8) / 4


async def test_reported_usage_is_used_as_is() -> None:
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    turn = await _client("ok", usage).complete(_MESSAGES, [])
    assert turn.usage == Usage(7, 3)
