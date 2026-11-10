"""Live LLM check: one real completion with one trivial tool. Skipped without credentials."""

from __future__ import annotations

import os

import pytest

from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.llm import OpenAICompatClient

pytestmark = [pytest.mark.live, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_real_completion_returns_a_tool_call() -> None:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    cfg = SandboxConfig.from_env()
    client = OpenAICompatClient(cfg.llm_base_url, cfg.llm_api_key.get_secret_value(), cfg.llm_model)
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a word back.",
                "parameters": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}},
                    "required": ["word"],
                },
            },
        }
    ]
    turn = await client.complete(
        [{"role": "user", "content": "Call the echo tool with the word 'hello'."}], tools
    )
    assert turn.tool_calls and turn.tool_calls[0].name == "echo"
    assert turn.usage.prompt_tokens > 0
