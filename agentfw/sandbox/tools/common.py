"""Shared helpers for sandbox MCP tool servers."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def ok(data: Any) -> dict[str, Any]:
    """Tool result payload: lists are wrapped as {"data": [...]} so item_count is extractable."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list | tuple):
        return {"data": list(data)}
    return {"data": data}


def tool_error(msg: str) -> ToolError:
    """An error the MCP client sees as `isError=True`."""
    return ToolError(msg)


def mark_fake(server: MCPServer, backend: object) -> MCPServer:
    """Set ``server.is_fake`` from the backend's ``is_fake`` marker (True only on Fake backends)."""
    setattr(server, "is_fake", bool(getattr(backend, "is_fake", False)))  # noqa: B010
    return server
