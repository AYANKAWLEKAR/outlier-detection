"""Salesforce MCP server: SOQL guard, sObject Collections chunking, budget, live."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from agentfw.core.patterns import is_id_like
from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.tools.salesforce_server import (
    FakeSalesforceBackend,
    LiveSalesforceBackend,
    build_server,
    validate_soql,
)
from tests.sandbox.tools.conftest import call, tool_names

COMPANY = build_company(0)


@pytest.fixture
def backend() -> FakeSalesforceBackend:
    return FakeSalesforceBackend(COMPANY)


@pytest.fixture
def server(backend):  # type: ignore[no-untyped-def]
    return build_server(backend)


@pytest.mark.anyio
async def test_tool_names(server) -> None:  # type: ignore[no-untyped-def]
    assert await tool_names(server) == {
        "run_soql",
        "get_record",
        "update_record",
        "mass_update",
        "create_task",
    }


def test_fake_ids_are_sf_ids_and_deterministic() -> None:
    a, b = FakeSalesforceBackend(COMPANY), FakeSalesforceBackend(COMPANY)
    ids = a.ids()
    assert ids == b.ids() and ids
    for rid in ids.values():
        assert len(rid) == 15 and is_id_like(rid)


@pytest.mark.parametrize(
    "q",
    [
        "DELETE FROM Contact",
        "UPDATE Contact SET Email = 'x'",
        "  delete from Contact",
        "/* c */ DELETE FROM Contact",
        "INSERT INTO Contact",
        "SELECT Id FROM Contact; DELETE FROM Contact",
        "SELECT Id FROM Contact LIMIT 2001",
        "SELECT Id FROM Contact LIMIT 99999999",
        "SELECT Id FROM Contact LIMIT :n",
        "",
    ],
)
def test_validate_soql_rejects(q: str) -> None:
    with pytest.raises(ToolError, match="(?i)soql|select|limit"):
        validate_soql(q)


def test_validate_soql_accepts_and_limits() -> None:
    assert validate_soql("SELECT Id FROM Contact").endswith("LIMIT 2000")
    assert validate_soql("\n select Id from Contact limit 5") == "select Id from Contact limit 5"
    assert validate_soql("SELECT Id FROM Contact WHERE Name = 'delete me'").endswith("LIMIT 2000")
    assert validate_soql("SELECT Id FROM Contact LIMIT 2000").endswith("LIMIT 2000")
    assert validate_soql("SELECT LastModifiedDate FROM Contact").endswith("LIMIT 2000")


@pytest.mark.parametrize(
    "q",
    [
        "SELECT Id FROM Account WHERE Website = 'http://a.example' AND Name = 'x'",
        "SELECT Id FROM Contact WHERE Name = 'a--b'",
        "SELECT Id FROM Contact WHERE Name = 'a /* b */ c' LIMIT 7",
        "SELECT Id, (SELECT Id FROM Contacts LIMIT 5) FROM Account",
        "SELECT Id, (SELECT Id FROM Contacts LIMIT 5) FROM Account LIMIT 10",
    ],
)
def test_validate_soql_preserves_original_text(q: str) -> None:
    out = validate_soql(q)
    assert out == q or out == f"{q} LIMIT 2000"


@pytest.mark.parametrize(
    "q",
    [
        "SELECT Id FROM Contact /* x */",
        "SELECT Id FROM Contact -- x",
        "SELECT Id FROM Contact // x",
        "SELECT Id, (SELECT Id FROM Contacts LIMIT 5000) FROM Account",
        "SELECT Id, (SELECT Id FROM Contacts LIMIT 5) FROM Account LIMIT 5000",
        "SELECT Id, (DELETE FROM Contacts) FROM Account",
        "SELECT Id FROM Contact WHERE Id IN (Delete x)",
        "SELECT Id FROM Contact WHERE Name = 'x' dElEtE",
        "SELECT Id FROM Contact WHERE Name = 'unterminated",
    ],
)
def test_validate_soql_rejects_more(q: str) -> None:
    with pytest.raises(ToolError):
        validate_soql(q)


@pytest.mark.anyio
async def test_reserved_keys_rejected_in_updates(server) -> None:  # type: ignore[no-untyped-def]
    _, res = await call(server, "run_soql", query="SELECT Id FROM Contact LIMIT 2")
    rids = [r["Id"] for r in res["data"]]
    bads: list[dict[str, Any]] = [
        {"Id": rids[0]},
        {"id": rids[0]},
        {"Attributes": {}},
        {"Title": "x", "ID": "y"},
    ]
    for bad in bads:
        err, msg = await call(
            server, "mass_update", object="Contact", record_ids=rids, field_updates=bad
        )
        assert err and "reserved" in msg.lower()
        err, msg = await call(
            server, "update_record", object="Contact", record_id=rids[0], fields=bad
        )
        assert err and "reserved" in msg.lower()


@pytest.mark.anyio
async def test_record_ids_validated(server) -> None:  # type: ignore[no-untyped-def]
    err, _ = await call(server, "get_record", object="Contact", record_id="../limits")
    assert err
    err, _ = await call(
        server, "mass_update", object="Contact", record_ids=["a/b"], field_updates={"Title": "x"}
    )
    assert err
    err, _ = await call(server, "create_task", what_id="x y", subject="s")
    assert err


@pytest.mark.anyio
async def test_mass_update_unknown_object_is_clean_error(server) -> None:  # type: ignore[no-untyped-def]
    err, msg = await call(
        server,
        "mass_update",
        object="Nope",
        record_ids=["003000000000001"],
        field_updates={"Title": "x"},
    )
    assert err and "unsupported object" in msg


@pytest.mark.anyio
async def test_mass_update_failure_reports_progress(tmp_path: Path) -> None:
    sf = _FakeSF()
    budget = ApiBudget(tmp_path / "b.json", limit=1)
    srv = build_server(LiveSalesforceBackend(_cfg(tmp_path), client=sf, budget=budget))
    ids = [f"003{i:012d}" for i in range(450)]
    err, msg = await call(
        srv, "mass_update", object="Contact", record_ids=ids, field_updates={"Title": "x"}
    )
    assert err and "updated=200" in msg and "batches=1" in msg


@pytest.mark.anyio
async def test_live_id_not_overridable(tmp_path: Path) -> None:
    sf = _FakeSF()
    budget = ApiBudget(tmp_path / "b.json", limit=5)
    srv = build_server(LiveSalesforceBackend(_cfg(tmp_path), client=sf, budget=budget))
    err, _ = await call(
        srv,
        "mass_update",
        object="Contact",
        record_ids=["003000000000001", "003000000000002"],
        field_updates={"Id": "003000000000009"},
    )
    assert err and sf.calls == []


@pytest.mark.anyio
async def test_soql_dml_is_error(server) -> None:  # type: ignore[no-untyped-def]
    err, msg = await call(server, "run_soql", query="DELETE FROM Contact")
    assert err and "select" in msg.lower()
    err, _ = await call(server, "run_soql", query="UPDATE Contact SET Email='a@b.example'")
    assert err


@pytest.mark.anyio
async def test_soql_select_and_get_record(server) -> None:  # type: ignore[no-untyped-def]
    err, res = await call(server, "run_soql", query="SELECT Id, Email FROM Contact LIMIT 3")
    assert not err and len(res["data"]) == 3
    rec = res["data"][0]
    assert is_id_like(rec["Id"]) and rec["Id"].startswith("003")
    err, got = await call(server, "get_record", object="Contact", record_id=rec["Id"])
    assert not err and got["Email"] == rec["Email"]
    err, _ = await call(server, "get_record", object="Contact", record_id="003000000000000")
    assert err
    err, _ = await call(server, "get_record", object="Nope", record_id=rec["Id"])
    assert err


@pytest.mark.anyio
async def test_update_record(server) -> None:  # type: ignore[no-untyped-def]
    _, res = await call(server, "run_soql", query="SELECT Id FROM Contact LIMIT 1")
    rid = res["data"][0]["Id"]
    err, _ = await call(
        server, "update_record", object="Contact", record_id=rid, fields={"Email": "n@x.example"}
    )
    assert not err
    _, got = await call(server, "get_record", object="Contact", record_id=rid)
    assert got["Email"] == "n@x.example"
    err, _ = await call(server, "update_record", object="Contact", record_id=rid, fields={})
    assert err


@pytest.mark.anyio
async def test_create_task(server) -> None:  # type: ignore[no-untyped-def]
    _, res = await call(server, "run_soql", query="SELECT Id FROM Account LIMIT 1")
    what = res["data"][0]["Id"]
    err, task = await call(server, "create_task", what_id=what, subject="Follow up")
    assert not err and task["id"].startswith("00T") and is_id_like(task["id"])
    err, _ = await call(server, "create_task", what_id="001000000000000", subject="x")
    assert err


@pytest.mark.anyio
async def test_mass_update_450_ids_makes_three_batches(backend) -> None:  # type: ignore[no-untyped-def]
    ids = [f"003{i:012d}" for i in range(450)]
    sizes: list[int] = []

    class Spy:
        def __getattr__(self, name: str) -> Any:
            return getattr(backend, name)

        def mass_update_batch(
            self, obj: str, records: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            sizes.append(len(records))
            return [{"id": r["Id"], "success": True} for r in records]

    srv = build_server(Spy())
    err, res = await call(
        srv, "mass_update", object="Contact", record_ids=ids, field_updates={"Title": "x"}
    )
    assert not err
    assert sizes == [200, 200, 50]
    assert res["updated"] == 450 and res["batches"] == 3


@pytest.mark.anyio
async def test_mass_update_fake_applies(server) -> None:  # type: ignore[no-untyped-def]
    _, res = await call(server, "run_soql", query="SELECT Id FROM Contact LIMIT 4")
    rids = [r["Id"] for r in res["data"]]
    err, out = await call(
        server, "mass_update", object="Contact", record_ids=rids, field_updates={"Title": "VIP"}
    )
    assert not err and out["updated"] == 4
    for rid in rids:
        _, got = await call(server, "get_record", object="Contact", record_id=rid)
        assert got["Title"] == "VIP"
    err, _ = await call(
        server, "mass_update", object="Contact", record_ids=[], field_updates={"Title": "x"}
    )
    assert err


class _FakeSF:
    """Stand-in for simple_salesforce.Salesforce: counts HTTP-ish calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def restful(self, path: str, params: Any = None, method: str = "GET", **kw: Any) -> Any:
        self.calls.append((method, path))
        if method == "PATCH":
            return [{"id": r["id"], "success": True} for r in orjson.loads(kw["data"])["records"]]
        return {"records": []}

    def query(self, q: str) -> dict[str, Any]:
        self.calls.append(("QUERY", q))
        return {"records": []}


def _cfg(tmp_path: Path) -> SandboxConfig:
    env = {n.upper(): "x" for n in SandboxConfig.REQUIRED_ENV}
    env.update(
        STRIPE_SECRET_KEY="sk_test_abc",
        LLM_USD_BUDGET="1",
        PRICE_IN_PER_M="1",
        PRICE_OUT_PER_M="1",
        DATA_DIR=str(tmp_path),
    )
    return SandboxConfig.from_env(env)


@pytest.mark.anyio
async def test_budget_overrun_is_error_and_charges_per_request(tmp_path: Path) -> None:
    sf = _FakeSF()
    budget = ApiBudget(tmp_path / "b.json", limit=2)
    srv = build_server(LiveSalesforceBackend(_cfg(tmp_path), client=sf, budget=budget))
    ids = [f"003{i:012d}" for i in range(450)]
    err, msg = await call(
        srv, "mass_update", object="Contact", record_ids=ids, field_updates={"Title": "x"}
    )
    assert err and "budget exceeded" in msg.lower()
    assert len(sf.calls) == 2  # third batch never sent
    err, msg = await call(srv, "run_soql", query="SELECT Id FROM Contact")
    assert err and "budget exceeded" in msg.lower()
    assert len(sf.calls) == 2


@pytest.mark.anyio
async def test_live_backend_rejects_dml_before_spending(tmp_path: Path) -> None:
    sf = _FakeSF()
    budget = ApiBudget(tmp_path / "b.json", limit=5)
    srv = build_server(LiveSalesforceBackend(_cfg(tmp_path), client=sf, budget=budget))
    err, _ = await call(srv, "run_soql", query="DELETE FROM Contact")
    assert err and sf.calls == [] and budget.remaining() == 5


def _live_cfg() -> SandboxConfig:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    return SandboxConfig.from_env()


@pytest.mark.live
@pytest.mark.anyio
async def test_live_soql_select() -> None:
    cfg = _live_cfg()
    srv = build_server(LiveSalesforceBackend(cfg))
    err, res = await call(srv, "run_soql", query="SELECT Id FROM Account LIMIT 1")
    assert not err and "data" in res
