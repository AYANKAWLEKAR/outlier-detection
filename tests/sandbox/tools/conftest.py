"""Shared helpers: drive an MCPServer through an in-memory MCP client."""

from __future__ import annotations

from typing import Any

import orjson
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def call(server: MCPServer, tool: str, **args: Any) -> tuple[bool, Any]:
    """Call `tool` over an in-memory client; return (is_error, parsed JSON or error text)."""
    async with Client(server) as client:
        res = await client.call_tool(tool, args)
    text = res.content[0].text  # type: ignore[union-attr]
    return res.is_error, (text if res.is_error else orjson.loads(text))


async def tool_names(server: MCPServer) -> set[str]:
    async with Client(server) as client:
        return {t.name for t in (await client.list_tools()).tools}
