"""`python -m agentfw.sandbox.proxy_py`: args and servers.yaml parsing, plus a stdio round trip."""

from __future__ import annotations

import sys
from pathlib import Path

import orjson
import pytest
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from agentfw.bench.dataset import load_traces
from agentfw.sandbox.proxy_py.__main__ import load_servers, parse_args

REPO = Path(__file__).resolve().parents[3]

SERVERS_YAML = """\
servers:
  zendesk:
    command: python
    args: [-m, agentfw.sandbox.tools.zendesk_server]
    env: {ZENDESK_TOKEN: hunter2-not-real}
  flaky_2:
    command: /usr/bin/env
    cwd: /tmp
"""


def test_load_servers(tmp_path: Path) -> None:
    p = tmp_path / "servers.yaml"
    p.write_text(SERVERS_YAML)
    cfg = load_servers(p)
    assert list(cfg.servers) == ["zendesk", "flaky_2"]
    zd = cfg.servers["zendesk"]
    assert zd.command == "python" and zd.args == ["-m", "agentfw.sandbox.tools.zendesk_server"]
    params = zd.to_params()
    assert params.env == {"ZENDESK_TOKEN": "hunter2-not-real"}
    assert "hunter2" not in repr(cfg)  # env values may be secrets
    assert cfg.servers["flaky_2"].to_params().env is None
    assert str(cfg.servers["flaky_2"].to_params().cwd) == "/tmp"


@pytest.mark.parametrize(
    "body",
    [
        "servers:\n  Zendesk: {command: x}\n",
        "servers:\n  zen-desk: {command: x}\n",
        "servers:\n  zen.desk: {command: x}\n",
        "servers: {}\n",
        "servers:\n  zendesk: {command: x, unknown: 1}\n",
        "- not a mapping\n",
    ],
)
def test_load_servers_rejects_bad_config(tmp_path: Path, body: str) -> None:
    p = tmp_path / "servers.yaml"
    p.write_text(body)
    with pytest.raises(ValueError):
        load_servers(p)


def test_parse_args() -> None:
    a = parse_args(
        ["--servers", "s.yaml", "--session-id", "sess_1", "--agent-id", "ag_1", "--out", "o"]
    )
    assert a.servers == Path("s.yaml") and a.out == Path("o")
    assert (a.session_id, a.agent_id, a.principal_id) == ("sess_1", "ag_1", "ag_1")
    assert a.timeout == 30.0
    b = parse_args(
        [
            "--servers=s.yaml",
            "--session-id=s",
            "--agent-id=a",
            "--principal-id=p",
            "--out=o",
            "--timeout=2.5",
        ]
    )
    assert (b.principal_id, b.timeout) == ("p", 2.5)
    with pytest.raises(SystemExit):
        parse_args(["--servers", "s.yaml", "--agent-id", "a", "--out", "o"])
    with pytest.raises(SystemExit):
        parse_args(["--servers=s", "--session-id=s", "--agent-id=a", "--out=o", "--timeout=0"])


@pytest.mark.anyio
async def test_stdio_round_trip(tmp_path: Path) -> None:
    """The real entrypoint, over stdio, in front of a stdio downstream (no network)."""
    servers = tmp_path / "servers.yaml"
    servers.write_text(
        f"servers:\n  flaky:\n    command: {sys.executable}\n"
        f"    args: [-m, tests.sandbox.proxy.flaky]\n    cwd: {REPO}\n"
    )
    out = tmp_path / "out"
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "agentfw.sandbox.proxy_py",
            "--servers",
            str(servers),
            "--session-id",
            "sess_stdio",
            "--agent-id",
            "agent_stdio",
            "--out",
            str(out),
        ],
        cwd=REPO,
    )
    async with Client(params) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"flaky.boom", "flaky.hang", "flaky.text", "flaky.ok"}
        res = await client.call_tool("flaky.ok", {})
        block = res.content[0]
        assert not res.is_error and isinstance(block, TextContent)
        assert len(orjson.loads(block.text)["data"]) == 2
        assert (await client.call_tool("flaky.boom", {})).is_error
    calls = load_traces(out / "calls.jsonl")
    assert [(c.tool_id, c.seq_no, c.session_id) for c in calls] == [
        ("flaky.ok", 0, "sess_stdio"),
        ("flaky.boom", 1, "sess_stdio"),
    ]
    assert calls[1].response_meta is not None and calls[1].response_meta.error is True
