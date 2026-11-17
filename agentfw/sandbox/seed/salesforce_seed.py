"""Seed a Salesforce Developer Edition org. Idempotent via Description prefix `afw-seed:<key>`,
falling back to each record's natural key (see ``_NATURAL``) when a task overwrote the marker."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.seed.company import Company
from agentfw.sandbox.tools.salesforce_server import require_developer_org

PREFIX = "afw-seed:"
_REFERENCE_DATE = (2026, 1, 1)


def _close_date(days: int) -> str:
    return (date(*_REFERENCE_DATE) + timedelta(days=days)).isoformat()


#: Fields that identify a seeded record when an agent task overwrote its ``Description`` marker.
_NATURAL: dict[str, tuple[str, ...]] = {
    "Account": ("Name", "Website"),
    "Contact": ("Email",),
    "Opportunity": ("AccountId", "Name"),
}


def _natural(kind: str, rec: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(rec.get(f) or "") for f in _NATURAL[kind])


class _Have:
    """Seeded records of one kind: by ``afw-seed:`` marker, else by natural key."""

    def __init__(self, kind: str, records: list[dict[str, Any]]) -> None:
        self.kind = kind
        self.by_key: dict[str, str] = {}
        self.by_natural: dict[tuple[str, ...], str] = {}
        for r in records:
            desc = r.get("Description") or ""
            if desc.startswith(PREFIX):
                self.by_key[desc[len(PREFIX) :]] = r["Id"]
            elif all(r.get(f) for f in _NATURAL[kind]):
                self.by_natural.setdefault(_natural(kind, r), r["Id"])

    def find(self, key: str, data: dict[str, Any]) -> str | None:
        if key in self.by_key:
            return self.by_key[key]
        return self.by_natural.get(_natural(self.kind, data))


def _existing(sf: Any, kind: str, budget: ApiBudget) -> _Have:
    budget.charge(1)
    # Description is a Long Text Area: not filterable in SOQL, so filter the prefix here.
    fields = ", ".join(("Id", "Description", *_NATURAL[kind]))
    res = sf.query_all(f"SELECT {fields} FROM {kind}")
    return _Have(kind, list(res["records"]))


def _create(
    sf: Any, kind: str, key: str, data: dict[str, Any], have: _Have, budget: ApiBudget
) -> str:
    found = have.find(key, data)
    if found is not None:
        have.by_key[key] = found
        return found
    budget.charge(1)
    rid = str(getattr(sf, kind).create({**data, "Description": f"{PREFIX}{key}"})["id"])
    have.by_key[key] = rid
    return rid


def seed_salesforce(
    cfg: SandboxConfig, company: Company, budget: ApiBudget, client: Any | None = None
) -> dict[str, str]:
    """Return {record key: Salesforce Id}; every API call is charged to `budget`. Refuses
    (``UnsafeConfigError``) a non-Developer-Edition, non-sandbox org before writing anything."""
    sf = client
    if sf is None:
        from simple_salesforce.api import Salesforce

        sf = Salesforce(
            username=cfg.sf_username,
            password=cfg.sf_password.get_secret_value(),
            security_token=cfg.sf_security_token.get_secret_value(),
            domain=cfg.sf_domain,
        )
    budget.charge(1)
    require_developer_org(sf)  # before any write: never seed a production org
    ids: dict[str, str] = {}
    have_acct = _existing(sf, "Account", budget)
    have_cont = _existing(sf, "Contact", budget)
    have_opp = _existing(sf, "Opportunity", budget)
    for a in company.accounts:
        ids[a.key] = _create(
            sf, "Account", a.key, {"Name": a.name, "Website": a.domain}, have_acct, budget
        )
    for c in company.contacts:
        ids[c.key] = _create(
            sf,
            "Contact",
            c.key,
            {
                "AccountId": ids[c.account_key],
                "FirstName": c.first_name,
                "LastName": c.last_name,
                "Email": c.email,
            },
            have_cont,
            budget,
        )
    for o in company.opportunities:
        ids[o.key] = _create(
            sf,
            "Opportunity",
            o.key,
            {
                "AccountId": ids[o.account_key],
                "Name": o.name,
                "StageName": o.stage,
                "Amount": o.amount,
                "CloseDate": _close_date(o.close_day),
            },
            have_opp,
            budget,
        )
    return ids
