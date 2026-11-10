"""Live seeding checks: skipped unless sandbox credentials are present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentfw.sandbox.config import ApiBudget, SandboxConfig
from agentfw.sandbox.seed.company import build_company
from agentfw.sandbox.seed.salesforce_seed import seed_salesforce
from agentfw.sandbox.seed.stripe_seed import seed_stripe

pytestmark = pytest.mark.live


def _cfg(*names: str) -> SandboxConfig:
    if any(n not in os.environ for n in names):
        pytest.skip(f"missing env: {names}")
    return SandboxConfig.from_env()


def test_stripe_seed_twice_no_duplicates() -> None:
    cfg = _cfg("STRIPE_SECRET_KEY", "LLM_BASE_URL")
    company = build_company(0)
    first = seed_stripe(cfg, company)
    second = seed_stripe(cfg, company)
    assert first == second


def test_salesforce_seed_twice_no_duplicates(tmp_path: Path) -> None:
    cfg = _cfg("SF_USERNAME", "SF_PASSWORD", "SF_SECURITY_TOKEN", "STRIPE_SECRET_KEY")
    company = build_company(0)
    budget = ApiBudget(tmp_path / "b.json", cfg.sf_daily_api_budget)
    assert seed_salesforce(cfg, company, budget) == seed_salesforce(cfg, company, budget)
