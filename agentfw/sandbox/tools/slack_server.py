"""Slack MCP server: only the one configured sandbox channel is ever posted to or read."""

from __future__ import annotations

import threading
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer

from agentfw.sandbox.config import SandboxConfig
from agentfw.sandbox.tools.common import ok, tool_error


class SlackBackend(Protocol):
    def post_message(self, channel: str, text: str) -> dict[str, Any]: ...
    def read_channel(self, limit: int) -> list[dict[str, Any]]: ...


def _check_channel(channel: str, allowed: str) -> None:
    if channel != allowed:
        raise tool_error(f"channel {channel!r} refused: only the sandbox channel is allowed")


def _check_limit(limit: int) -> int:
    if limit < 1:
        raise tool_error("limit must be >= 1")
    return min(limit, 100)


class FakeSlackBackend:
    def __init__(self, channel_id: str) -> None:
        self._channel = channel_id
        self._msgs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def post_message(self, channel: str, text: str) -> dict[str, Any]:
        _check_channel(channel, self._channel)
        with self._lock:
            ts = f"{1_700_000_000 + len(self._msgs)}.000100"
            self._msgs.append({"ts": ts, "user": "UBOT", "text": text})
        return {"ok": True, "channel": channel, "ts": ts}

    def read_channel(self, limit: int) -> list[dict[str, Any]]:
        n = _check_limit(limit)
        with self._lock:
            return list(reversed(self._msgs))[:n]


class LiveSlackBackend:
    def __init__(self, channel_id: str, client: Any) -> None:
        self._channel = channel_id
        self._client = client

    @classmethod
    def from_config(cls, cfg: SandboxConfig) -> LiveSlackBackend:
        from slack_sdk import WebClient

        return cls(cfg.slack_channel_id, WebClient(token=cfg.slack_bot_token.get_secret_value()))

    def post_message(self, channel: str, text: str) -> dict[str, Any]:
        _check_channel(channel, self._channel)
        try:
            res = self._client.chat_postMessage(channel=channel, text=text)
        except Exception as exc:
            raise tool_error(f"Slack error: {exc}") from exc
        return {"ok": bool(res["ok"]), "channel": channel, "ts": res["ts"]}

    def read_channel(self, limit: int) -> list[dict[str, Any]]:
        n = _check_limit(limit)
        try:
            res = self._client.conversations_history(channel=self._channel, limit=n)
        except Exception as exc:
            raise tool_error(f"Slack error: {exc}") from exc
        return [
            {"ts": m.get("ts"), "user": m.get("user"), "text": m.get("text", "")}
            for m in res["messages"]
        ]


def build_server(backend: SlackBackend) -> MCPServer:
    server = MCPServer("slack")

    @server.tool()
    def post_message(channel: str, text: str) -> dict[str, Any]:
        """Post a message to the sandbox channel."""
        return ok(backend.post_message(channel, text))

    @server.tool()
    def read_channel(limit: int = 20) -> dict[str, Any]:
        """Read recent messages from the sandbox channel (newest first)."""
        return ok(backend.read_channel(limit))

    return server
