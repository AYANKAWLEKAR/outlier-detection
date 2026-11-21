"""Shared helpers for sandbox MCP tool servers."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations


def _ann(read_only: bool, destructive: bool, idempotent: bool, open_world: bool) -> ToolAnnotations:
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=open_world,
    )


# MCP annotations (F3). They agree with the synth catalog's (`agentfw/synth/catalog/*.yaml`; a
# test compares them). Reads are idempotent on purpose: T-26's polling rule reads the hint.
READ = _ann(True, False, True, False)  # no side effect
ADD_CLOSED = _ann(False, False, False, False)  # a new record per call, inside the account
ADD_OPEN = _ann(False, False, False, True)  # additive, but reaches parties outside
OVERWRITE = _ann(False, True, True, False)  # sets a field to a value (idempotent)
DESTRUCTIVE_OPEN = _ann(False, True, False, True)  # moves money out of the account


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
