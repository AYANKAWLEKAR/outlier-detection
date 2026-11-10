"""Seeder idempotency against in-memory fakes (no network)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from agentfw.sandbox.config import ApiBudget, BudgetExceeded, SandboxConfig, UnsafeConfigError
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.seed.salesforce_seed import seed_salesforce
from agentfw.sandbox.seed.stripe_seed import seed_stripe
from agentfw.sandbox.seed.zendesk_seed import seed_zendesk


def _cfg(tmp_path: Path) -> SandboxConfig:
    return SandboxConfig(
        llm_base_url="https://llm.example/v1",
        llm_api_key="x",
        llm_model="m",
        judge_base_url="https://j.example/v1",
        judge_api_key="x",
        judge_model="m",
        stripe_secret_key="sk_test_fake",
        sf_username="u@x.example",
        sf_password="x",
        sf_security_token="x",
        slack_bot_token="x",
        slack_channel_id="C1",
        llm_usd_budget=1.0,
        price_in_per_m=1.0,
        price_out_per_m=1.0,
        data_dir=tmp_path,
    )


class _Page(list[Any]):
    def auto_paging_iter(self) -> Any:
        return iter(list(self))


class _Resource:
    def __init__(self, prefix: str, store: list[dict[str, Any]]) -> None:
        self.prefix = prefix
        self.store = store
        self.status = "succeeded"
        self.creates: list[dict[str, Any]] = []

    def list(self, params: dict[str, Any] | None = None) -> _Page:
        p = params or {}
        out = [
            o
            for o in self.store
            if all(o.get(k) == v for k, v in p.items() if k in ("email", "customer"))
        ]
        return _Page(SimpleNamespace(**o) for o in out)

    def create(self, params: dict[str, Any]) -> Any:
        obj = dict(params)
        obj["id"] = f"{self.prefix}_{len(self.store):04d}"
        if self.prefix == "pi":
            obj["latest_charge"] = f"ch_{len(self.store):04d}"
            obj["status"] = self.status
        self.store.append(obj)
        self.creates.append(params)
        return SimpleNamespace(**obj)


class _FakeStripe:
    def __init__(self) -> None:
        self.customers = _Resource("cus", [])
        self.payment_intents = _Resource("pi", [])


def test_stripe_seed_idempotent_and_params(tmp_path: Path) -> None:
    cfg, company, api = _cfg(tmp_path), build_company(0), _FakeStripe()
    first = seed_stripe(cfg, company, client=api)
    n_cus, n_pi = len(api.customers.store), len(api.payment_intents.store)
    assert n_cus == len(company.customers) and n_pi == len(company.charges)
    assert all(first[c.key].startswith("cus_") for c in company.customers)
    assert all(first[c.key].startswith("ch_") for c in company.charges)
    second = seed_stripe(cfg, company, client=api)
    assert second == first
    assert len(api.customers.store) == n_cus and len(api.payment_intents.store) == n_pi
    for p in api.payment_intents.creates:
        assert p["payment_method"] == "pm_card_bypassPending"
        assert p["confirm"] is True
        assert p["automatic_payment_methods"] == {"enabled": True, "allow_redirects": "never"}
        assert p["metadata"]["afw_seed_key"]


def test_stripe_seed_fails_on_unsucceeded_payment(tmp_path: Path) -> None:
    api = _FakeStripe()
    api.payment_intents.status = "requires_action"
    with pytest.raises(RuntimeError, match="requires_action"):
        seed_stripe(_cfg(tmp_path), build_company(0), client=api)


def test_stripe_seed_fails_on_missing_charge(tmp_path: Path) -> None:
    api = _FakeStripe()
    orig = api.payment_intents.create

    def create(params: dict[str, Any]) -> Any:
        pi = orig(params)
        pi.latest_charge = None
        return pi

    api.payment_intents.create = create  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="charge"):
        seed_stripe(_cfg(tmp_path), build_company(0), client=api)


def test_stripe_seed_refuses_unsafe_key(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path).model_copy(update={"stripe_secret_key": SecretStr("sk_live_zzz")})
    with pytest.raises(UnsafeConfigError):
        seed_stripe(cfg, build_company(0), client=_FakeStripe())


class _FakeSObject:
    def __init__(self, kind: str, sf: _FakeSF) -> None:
        self.kind, self.sf = kind, sf

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        rid = f"{self.kind[:3]}{len(self.sf.records[self.kind]):05d}"
        self.sf.records[self.kind].append({"Id": rid, **data})
        return {"id": rid, "success": True}


class _FakeSF:
    def __init__(self) -> None:
        self.records: dict[str, list[dict[str, Any]]] = {
            "Account": [],
            "Contact": [],
            "Opportunity": [],
        }
        self.queries = 0
        for k in self.records:
            setattr(self, k, _FakeSObject(k, self))

    def query_all(self, soql: str) -> dict[str, Any]:
        self.queries += 1
        where = soql.split(" WHERE ")[1] if " WHERE " in soql else ""
        assert "Description" not in where, "Description (long text) is not filterable in SOQL"
        kind = soql.split(" FROM ")[1].split()[0]
        return {"records": list(self.records[kind])}


def test_salesforce_seed_idempotent(tmp_path: Path) -> None:
    cfg, company, sf = _cfg(tmp_path), build_company(0), _FakeSF()
    budget = ApiBudget(tmp_path / "b.json", limit=100_000)
    first = seed_salesforce(cfg, company, budget, client=sf)
    total = sum(len(v) for v in sf.records.values())
    assert total == company.salesforce_record_count()
    assert all(r["Description"].startswith("afw-seed:") for v in sf.records.values() for r in v)
    used = 100_000 - budget.remaining()
    assert used >= total
    second = seed_salesforce(cfg, company, budget, client=sf)
    assert second == first
    assert sum(len(v) for v in sf.records.values()) == total
    assert 100_000 - budget.remaining() - used <= 3  # only the three existence queries


def test_salesforce_seed_respects_budget(tmp_path: Path) -> None:
    budget = ApiBudget(tmp_path / "b.json", limit=5)
    with pytest.raises(BudgetExceeded):
        seed_salesforce(_cfg(tmp_path), build_company(0), budget, client=_FakeSF())


def test_zendesk_seed(tmp_path: Path) -> None:
    company = build_company(0)
    ids = {c.key: f"cus_{c.key}" for c in company.customers}
    ids.update({c.key: f"ch_{c.key}" for c in company.charges})
    db = tmp_path / "zd.sqlite"
    seed_zendesk(db, company, ids)
    seed_zendesk(db, company, ids)  # idempotent
    rows = (
        sqlite3.connect(db).execute("SELECT key, charge_id, customer_email FROM tickets").fetchall()
    )
    assert len(rows) == len(company.tickets) > 0
    assert all(r[1].startswith("ch_") and r[2].endswith((".example", ".test")) for r in rows)


def test_salesforce_seed_ignores_foreign_records(tmp_path: Path) -> None:
    sf = _FakeSF()
    sf.records["Account"] += [
        {"Id": "foreign1", "Description": None},
        {"Id": "foreign2", "Description": "someone else's"},
    ]
    budget = ApiBudget(tmp_path / "b.json", limit=100_000)
    ids = seed_salesforce(_cfg(tmp_path), build_company(0), budget, client=sf)
    assert "foreign1" not in ids.values() and "foreign2" not in ids.values()
    assert len(sf.records["Account"]) == 2 + len(build_company(0).accounts)
