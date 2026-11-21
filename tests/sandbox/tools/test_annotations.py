"""Every sandbox tool is annotated (F3), and the annotations agree with the synth catalog's."""

from __future__ import annotations

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from agentfw.core.hints import ToolAnnotations
from agentfw.sandbox.backends import BUILDERS, fake_backends
from agentfw.sandbox.seed.company import build_company
from agentfw.synth.load import load_catalog

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _tools() -> dict[str, ToolAnnotations]:
    backends = fake_backends(build_company(0), "C0")
    out: dict[str, ToolAnnotations] = {}
    for name, be in backends.items():
        server: MCPServer = BUILDERS[name](be)
        async with Client(server) as client:
            for t in (await client.list_tools()).tools:
                a = t.annotations
                assert a is not None, f"{name}.{t.name} has no annotations"
                assert a.read_only_hint is not None and a.destructive_hint is not None, t.name
                out[f"{name}.{t.name}"] = ToolAnnotations(
                    read_only=a.read_only_hint,
                    destructive=a.destructive_hint,
                    idempotent=a.idempotent_hint,
                    open_world=a.open_world_hint,
                )
    return out


async def test_every_server_tool_is_annotated() -> None:
    assert len(await _tools()) == 17


async def test_server_annotations_agree_with_the_synth_catalog() -> None:
    served = await _tools()
    catalog = load_catalog()
    assert set(served) == set(catalog)
    for tool, spec in catalog.items():
        assert spec.annotations is not None
        assert served[tool] == spec.annotations.to_core(), tool
