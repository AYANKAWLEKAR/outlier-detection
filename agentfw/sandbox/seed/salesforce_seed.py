"""Seed a Salesforce Developer Edition org. Idempotent via Description prefix `afw-seed:<key>`."""

from __future__ import annotations

from typing import Any

from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.seed.company import Company

PREFIX = "afw-seed:"
_REFERENCE_DATE = (2026, 1, 1)


def _close_date(days: int) -> str:
    from datetime import date, timedelta

    return (date(*_REFERENCE_DATE) + timedelta(days=days)).isoformat()


def _existing(sf: Any, kind: str, budget: ApiBudget) -> dict[str, str]:
    budget.charge(1)
    res = sf.query_all(f"SELECT Id, Description FROM {kind} WHERE Description LIKE '{PREFIX}%'")
    return {r["Description"][len(PREFIX) :]: r["Id"] for r in res["records"]}


def _create(
    sf: Any, kind: str, key: str, data: dict[str, Any], have: dict[str, str], budget: ApiBudget
) -> str:
    if key in have:
        return have[key]
    budget.charge(1)
    rid = str(getattr(sf, kind).create({**data, "Description": f"{PREFIX}{key}"})["id"])
    have[key] = rid
    return rid


def seed_salesforce(
    cfg: SandboxConfig, company: Company, budget: ApiBudget, client: Any | None = None
) -> dict[str, str]:
    """Return {record key: Salesforce Id}; every API call is charged to `budget`."""
    sf = client
    if sf is None:
        from simple_salesforce.api import Salesforce

        sf = Salesforce(
            username=cfg.sf_username,
            password=cfg.sf_password,
            security_token=cfg.sf_security_token,
            domain=cfg.sf_domain,
        )
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
