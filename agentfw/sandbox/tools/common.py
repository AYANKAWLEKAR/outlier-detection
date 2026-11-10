"""Shared helpers for sandbox MCP tool servers."""

from __future__ import annotations

from typing import Any

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
