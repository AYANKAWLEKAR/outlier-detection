"""Live seeding checks: skipped unless sandbox credentials are present."""

from __future__ import annotations

import os

import pytest

from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.seed.salesforce_seed import seed_salesforce
from agentfw.sandbox.seed.stripe_seed import seed_stripe

pytestmark = pytest.mark.live


def _cfg() -> SandboxConfig:
    missing = [n.upper() for n in SandboxConfig.REQUIRED_ENV if n.upper() not in os.environ]
    if missing:
        pytest.skip(f"missing env: {missing}")
    return SandboxConfig.from_env()


def test_stripe_seed_twice_no_duplicates() -> None:
    cfg = _cfg()
    company = build_company(0)
    first = seed_stripe(cfg, company)
    second = seed_stripe(cfg, company)
    assert first == second


def test_salesforce_seed_twice_no_duplicates() -> None:
    cfg = _cfg()
    company = build_company(0)
    budget = ApiBudget(cfg.data_dir / "sf_budget.json", cfg.sf_daily_api_budget)
    assert seed_salesforce(cfg, company, budget) == seed_salesforce(cfg, company, budget)
