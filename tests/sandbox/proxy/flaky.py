"""A misbehaving MCP server (raises, hangs, returns non-JSON); runnable over stdio."""

from __future__ import annotations

from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

NOT_JSON = "plain text, définitivement not JSON"


def build_flaky() -> MCPServer:
    server = MCPServer("flaky")

    @server.tool()
    def boom() -> dict[str, Any]:
        """Always raises."""
        raise RuntimeError("kaboom")

    @server.tool()
    async def hang() -> dict[str, Any]:
        """Never answers in time."""
        await anyio.sleep(30)
        return {}

    @server.tool()
    def text() -> str:
        """Returns non-JSON text."""
        return NOT_JSON

    @server.tool()
    def ok() -> dict[str, Any]:
        """A healthy tool."""
        return {"data": [{"id": "ch_0000000001"}, {"id": "ch_0000000002"}]}

    return server


if __name__ == "__main__":
    build_flaky().run()
