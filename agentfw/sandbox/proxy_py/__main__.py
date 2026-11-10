"""Run the proxy over stdio in front of stdio downstream servers.

    python -m agentfw.sandbox.proxy_py --servers servers.yaml --session-id S --agent-id A --out DIR

``servers.yaml``::

    servers:
      zendesk:                       # lowercase snake: becomes the tool-id namespace
        command: python
        args: [-m, some.module]
        env: {KEY: value}            # optional; values are kept secret in reprs
        cwd: /path                   # optional

Envelopes go to ``DIR/{calls,labels,verdicts}.jsonl`` (``JsonlSink``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path

import anyio
import yaml
from mcp import Client, StdioServerParameters
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from agentfw.sandbox.proxy_py.proxy import NAME_PART, Downstream, FirewallProxy, SessionInfo
from agentfw.sandbox.proxy_py.sinks import JsonlSink

__all__ = ["DownstreamSpec", "ServersConfig", "load_servers", "main", "parse_args", "serve"]


class DownstreamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, SecretStr] | None = None
    cwd: Path | None = None

    def to_params(self) -> StdioServerParameters:
        env = None if self.env is None else {k: v.get_secret_value() for k, v in self.env.items()}
        return StdioServerParameters(command=self.command, args=self.args, env=env, cwd=self.cwd)


class ServersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    servers: dict[str, DownstreamSpec] = Field(min_length=1)

    @field_validator("servers")
    @classmethod
    def _names(cls, v: dict[str, DownstreamSpec]) -> dict[str, DownstreamSpec]:
        for name in v:
            if not NAME_PART.fullmatch(name):
                raise ValueError(f"server name {name!r} must be lowercase snake ([a-z0-9_]+)")
        return v


def load_servers(path: Path) -> ServersConfig:
    return ServersConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def _positive(s: str) -> float:
    v = float(s)
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {s}")
    return v


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m agentfw.sandbox.proxy_py", description=__doc__)
    p.add_argument("--servers", type=Path, required=True, help="servers.yaml")
    p.add_argument("--session-id", required=True)
    p.add_argument("--agent-id", required=True)
    p.add_argument("--principal-id", default=None, help="defaults to --agent-id")
    p.add_argument("--out", type=Path, required=True, help="envelope output directory")
    p.add_argument("--timeout", type=_positive, default=30.0, help="per-call timeout (s)")
    args = p.parse_args(argv)
    if args.principal_id is None:
        args.principal_id = args.agent_id
    return args


async def serve(args: argparse.Namespace) -> None:
    cfg = load_servers(args.servers)
    async with AsyncExitStack() as stack:
        downstreams: dict[str, Downstream] = {}
        for name, spec in cfg.servers.items():
            downstreams[name] = await stack.enter_async_context(Client(spec.to_params()))
        proxy = FirewallProxy(
            downstreams,
            SessionInfo(args.session_id, args.agent_id, args.principal_id),
            JsonlSink(args.out),
            timeout_s=args.timeout,
        )
        await proxy.list_tools()
        server = proxy.build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


def main(argv: Sequence[str] | None = None) -> int:
    anyio.run(serve, parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
